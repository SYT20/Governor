"""The episode loop: candidates -> admissibility -> choice -> execute -> record.

This is the harness every arm plugs into. It is deliberately the same code path for
the fixed baseline, the heuristic, the random collector, and Governor itself --
otherwise an arm comparison measures harness differences instead of policy
differences (drift check #6, arm parity).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Protocol

from governor.accounting.meter import (
    DIMENSIONS,
    Accountant,
    BudgetExhausted,
    Envelope,
)
from governor.cognitive.belief import Belief, Channel, entropy, update_belief
from governor.envs.synthbug import Action, Mode, Observation, SynthBug, Tier
from governor.models.cost import CostProfile, Reserve
from governor.record.store import DecisionRow, EpisodeRow, RecordStore


@dataclass(slots=True)
class EpisodeContext:
    """Everything an arm is allowed to see. Never contains the true cause."""

    belief: Belief
    accountant: Accountant
    profile: CostProfile
    n_hypotheses: int
    step: int
    last_test: int | None = None
    n_by_mode: dict[str, int] = field(default_factory=dict)
    steps_since_new_evidence: int = 0
    verified: bool = False

    # -- features consumed by the value model (all deterministic, section G) ---

    def features(self) -> dict[str, float]:
        b = self.belief
        return {
            "frac_budget_remaining": self.accountant.fraction_remaining(),
            "belief_entropy": entropy(b),
            "max_belief": max(b),
            "belief_margin": _margin(b),
            "n_explore": float(self.n_by_mode.get("EXPLORE", 0)),
            "n_exploit": float(self.n_by_mode.get("EXPLOIT", 0)),
            "n_verify": float(self.n_by_mode.get("VERIFY", 0)),
            "steps_since_new_evidence": float(self.steps_since_new_evidence),
            "last_test_pass": float(self.last_test) if self.last_test is not None else -1.0,
            "step": float(self.step),
            "n_hypotheses": float(self.n_hypotheses),
        }


def _margin(b: Belief) -> float:
    """Gap between the top two hypotheses. A cheap, robust confidence proxy."""
    if len(b) < 2:
        return 1.0
    s = sorted(b, reverse=True)
    return s[0] - s[1]


class Arm(Protocol):
    """A policy. The only contract is: given a context and legal actions, choose one."""

    name: str

    def reset(self, task: SynthBug) -> None: ...

    def act(self, ctx: EpisodeContext, admissible: list[Action]) -> tuple[Action, str]:
        """Return (action, reason_code)."""
        ...


def propose(arm: Arm, ctx: EpisodeContext) -> list[Action]:
    """The second half of the dual-source candidate set (section F.3).

    An arm may contribute semantic candidates the deterministic generator cannot
    know to offer -- this is where the LLM proposer plugs in, and where the oracle
    contributes its truth-targeted repair. Arms without a `propose` method simply
    add nothing.
    """
    fn = getattr(arm, "propose", None)
    return list(fn(ctx)) if callable(fn) else []


@dataclass(slots=True)
class EpisodeResult:
    episode_id: str
    arm: str
    seed: int
    succeeded: bool
    terminal: str | None
    n_decisions: int
    consumed: dict[str, float]
    truncations: int
    violated: bool
    ground_truth: dict[str, object]


def config_hash(*parts: object) -> str:
    """Stamp identifying the exact configuration an episode ran under.

    Drift check #6: every arm in a comparison must share this hash for the shared
    components (environment config, envelope, channel). If it differs, the
    comparison is confounded and the analysis should refuse to run.
    """
    blob = json.dumps([repr(p) for p in parts], sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def deterministic_candidates(ctx: EpisodeContext, tiers: tuple[Tier, ...]) -> list[Action]:
    """The completeness guarantee of Decision Record section F.3.

    Rev 1 let the LLM propose every candidate, which meant the controller could only
    ever choose from the model's brainstorm. This set is always present regardless of
    what any proposer suggests, so the policy can never be denied an obviously
    correct move.
    """
    acts: list[Action] = []
    order = sorted(range(ctx.n_hypotheses), key=lambda i: -ctx.belief[i])
    for tier in tiers:
        acts.append(Action(Mode.VERIFY, tier))
        # Probe *and* repair against the leading two hypotheses. Offering EXPLOIT
        # only against the argmax was a real completeness bug: it made a
        # truth-targeted repair unreachable whenever belief happened to lead
        # elsewhere, which silently capped the achievable ceiling.
        for h in order[:2]:
            acts.append(Action(Mode.EXPLORE, tier, h))
            acts.append(Action(Mode.EXPLOIT, tier, h))
    acts.append(Action(Mode.STOP_VERIFIED if ctx.verified else Mode.STOP_UNVERIFIED))
    acts.append(Action(Mode.STOP_FAILURE))
    return acts


def _charge_or_truncate(
    acc: Accountant, label: str, cost: dict[str, float]
) -> tuple[dict[str, float], bool]:
    """Charge the realised cost, clamping at the envelope.

    This is where the section H.1 distinction becomes visible. Admissibility used
    the p90 and can be wrong; when it is, the executor is killed at its hard cap and
    the event is recorded as a *truncation*. The envelope is never breached, so BVR
    stays 0 by construction while the truncation rate carries the information about
    how good the cost estimates were.
    """
    rem = acc.remaining()
    clamped = {d: min(cost.get(d, 0.0), max(0.0, rem[d])) for d in DIMENSIONS}
    truncated = any(clamped[d] < cost.get(d, 0.0) - 1e-9 for d in DIMENSIONS)
    acc.charge(label, truncated=truncated, **clamped)
    return clamped, truncated


def run_episode(
    *,
    task: SynthBug,
    arm: Arm,
    envelope: Envelope,
    profile: CostProfile,
    channel_for: dict[Tier, Channel],
    store: RecordStore | None = None,
    episode_id: str | None = None,
    budget_scale: float = 1.0,
    cfg_hash: str = "",
    tiers: tuple[Tier, ...] = tuple(Tier),
    max_steps: int = 40,
) -> EpisodeResult:
    """Run one episode of one arm against one task."""
    acc = Accountant(envelope=envelope)
    reserve = Reserve(profile)
    ctx = EpisodeContext(
        belief=task.prior,
        accountant=acc,
        profile=profile,
        n_hypotheses=task.config.n_hypotheses,
        step=0,
    )
    arm.reset(task)
    eid = episode_id or f"{arm.name}-{task.seed}-{budget_scale}"
    decisions = 0

    while not task.terminated and ctx.step < max_steps:
        det = deterministic_candidates(ctx, tiers)
        extra = propose(arm, ctx)
        source = {a: "deterministic" for a in det}
        for a in extra:
            source.setdefault(a, "proposed")
        candidates = list(source)
        need = reserve.required(verified=ctx.verified)

        admissible: list[Action] = []
        cand_records: list[dict[str, object]] = []
        for a in candidates:
            vec = profile.vector(a.action_class)
            res = {} if a.mode.is_terminal else need
            ok = acc.admissible(vec, reserve=res)
            if ok:
                admissible.append(a)
            cand_records.append(
                {
                    "action": str(a),
                    "class": a.action_class,
                    "admissible": ok,
                    "cost_p50": round(vec["cost"].value, 6),
                    "cost_p90": round(vec["cost"].ucb, 6),
                    "estimate_source": vec["cost"].source,
                    # Which generator offered this action. Section F.3: if the
                    # chosen action is always "deterministic", the proposer is dead
                    # weight; if always "proposed", the completeness guarantee is
                    # doing nothing and the oracle-scope caveat bites harder.
                    "provenance": source[a],
                }
            )

        if not admissible:
            # Nothing fits. Terminating is always permitted -- refusing to let the
            # episode close would be the one way to actually breach the envelope.
            admissible = [
                Action(Mode.STOP_VERIFIED if ctx.verified else Mode.STOP_UNVERIFIED)
            ]

        try:
            acc.dispatch_or_refuse({"tokens": 1.0})
        except BudgetExhausted:
            admissible = [Action(Mode.STOP_UNVERIFIED)]

        action, reason = arm.act(ctx, admissible)
        belief_before = list(ctx.belief)

        realised = task.cost_of(action)
        charged, truncated = _charge_or_truncate(acc, action.action_class, realised)
        if not profile.frozen:
            # Corpus collection only. During evaluation the profile is frozen, so
            # every arm is scored by the identical model (section L).
            profile.observe(action.action_class, realised)

        obs: Observation = task.step(action)
        _apply_observation(ctx, action, obs, channel_for, truncated)

        if store is not None:
            store.write_decision(
                DecisionRow(
                    episode_id=eid,
                    decision_id=decisions,
                    features=ctx.features(),
                    candidates=cand_records,
                    chosen=str(action),
                    reason_code=reason,
                    was_random=reason == "EXPLORATION_RANDOM",
                    epsilon=getattr(arm, "epsilon", 0.0),
                    provenance=source.get(action, "deterministic"),
                    actual_cost=charged,
                    observation={"kind": obs.kind, "value": obs.value, "truncated": truncated},
                    belief_before=belief_before,
                    belief_after=list(ctx.belief),
                    entropy_after=entropy(ctx.belief),
                )
            )
        decisions += 1
        ctx.step += 1

    acc.reconcile()  # drift check #2, every episode

    result = EpisodeResult(
        episode_id=eid,
        arm=arm.name,
        seed=task.seed,
        succeeded=task.succeeded(),
        terminal=str(task.terminal_mode) if task.terminal_mode else None,
        n_decisions=decisions,
        consumed=acc.consumed(),
        truncations=acc.truncations,
        violated=acc.violated(),
        ground_truth=task.ground_truth(),
    )
    if store is not None:
        store.write_episode(
            EpisodeRow(
                episode_id=eid,
                arm=arm.name,
                seed=task.seed,
                envelope=envelope.as_dict(),
                budget_scale=budget_scale,
                config_hash=cfg_hash,
                succeeded=result.succeeded,
                terminal=result.terminal,
                n_decisions=decisions,
                consumed=result.consumed,
                truncations=result.truncations,
                violated=result.violated,
                ground_truth=result.ground_truth,
            )
        )
    return result


def _apply_observation(
    ctx: EpisodeContext,
    action: Action,
    obs: Observation,
    channel_for: dict[Tier, Channel],
    truncated: bool,
) -> None:
    """Fold an observation into the context. The only writer of ctx.belief."""
    ctx.n_by_mode[str(action.mode)] = ctx.n_by_mode.get(str(action.mode), 0) + 1

    if truncated:
        # A killed action produced no usable signal.
        ctx.steps_since_new_evidence += 1
        return

    if obs.kind == "evidence" and obs.target is not None and obs.value is not None:
        ctx.belief = update_belief(
            ctx.belief, channel_for[action.tier], obs.target, obs.value
        )
        ctx.steps_since_new_evidence = 0
    elif obs.kind == "test" and obs.value is not None:
        ctx.last_test = obs.value
        ctx.verified = bool(obs.value)
        ctx.steps_since_new_evidence = 0
    else:
        ctx.steps_since_new_evidence += 1
