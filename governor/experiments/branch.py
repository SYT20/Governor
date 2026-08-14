"""Branched counterfactuals: does the model rank ACTIONS, or just rank STATES?

Stage 3 reported AUC ~0.84 for Q(s,a) and it was over-read -- including by me --
as evidence that action selection is sound. It is not.

Pooled AUC is computed over every (state, action) row at once, so it is dominated
by state-to-state variation: a model that perfectly separates easy episodes from
hard ones while ordering actions at random can score exactly that. The policy's
actual question is narrower and harder:

    at ONE state s, is Q(s, A) > Q(s, B) when A really is the better action?

Answering it requires counterfactuals -- the same state, different actions,
observed outcomes. On a real benchmark that means forking containers (The Replay
Gap's protocol, and the reason it is expensive). SynthBug is a simulator, so the
same experiment costs microseconds: deep-copy the task, force the candidate
action, roll forward, repeat.

Metrics produced here:

    pairwise ranking accuracy   over action pairs at a fixed state, the fraction
                                where predicted order matches realised order
    Kendall tau                 rank correlation between predicted and realised
                                action values at a state
    decision regret             realised value of the best tested action minus
                                that of the action the model would have chosen

Pairwise accuracy is the headline. A value of 0.5 means the model cannot rank
actions at all, whatever its AUC says.
"""

from __future__ import annotations

import copy
import statistics
from dataclasses import dataclass, field

from governor.accounting.meter import Accountant, Envelope
from governor.arms.baselines import HeuristicArm
from governor.cognitive.belief import Channel
from governor.corpus.build import Checkpoint
from governor.envs.families import Family
from governor.envs.synthbug import Action, Mode, SynthBug, Tier
from governor.models.cost import CostProfile, Reserve
from governor.policy.runner import (
    EpisodeContext,
    _apply_observation,
    _charge_or_truncate,
    deterministic_candidates,
)


@dataclass(slots=True)
class BranchPoint:
    """One state, every candidate action tried from it, with realised outcomes."""

    family: str
    split: str
    episode_seed: int
    decision_id: int
    features: dict[str, float]
    realised: dict[str, float] = field(default_factory=dict)   # action -> success rate
    n_replicates: int = 0
    predicted: dict[str, float] = field(default_factory=dict)  # action -> Q(s,a)

    def checkpoint_for(self, action: Action) -> Checkpoint:
        """Shape this state + a candidate action as a Checkpoint, so the fitted
        model can score it through exactly the same path it was trained on."""
        return Checkpoint(
            episode_id=f"branch|{self.family}|{self.episode_seed}|{self.decision_id}",
            decision_id=self.decision_id,
            family=self.family,
            split=self.split,
            seed=self.episode_seed,
            action=str(action),
            mode=str(action.mode),
            tier=str(action.tier),
            was_random=False,
            n_admissible=len(self.realised) or 1,
            features=dict(self.features),
            label=0,
        )


def _fresh_context(task: SynthBug, acc: Accountant, profile: CostProfile) -> EpisodeContext:
    return EpisodeContext(
        belief=task.prior, accountant=acc, profile=profile,
        n_hypotheses=task.config.n_hypotheses, step=0,
    )


def _rollout(
    task: SynthBug,
    ctx: EpisodeContext,
    profile: CostProfile,
    channels: dict[Tier, Channel],
    tiers: tuple[Tier, ...],
    max_steps: int,
) -> bool:
    """Continue to termination under a fixed policy, and report success.

    The continuation policy is held constant across every branch so that the only
    thing differing between arms is the forced first action. Otherwise the
    comparison measures the rollout policy, not the action.
    """
    arm = HeuristicArm()
    arm.reset(task)
    reserve = Reserve(profile)
    while not task.terminated and ctx.step < max_steps:
        cands = deterministic_candidates(ctx, tiers)
        need = reserve.required(verified=ctx.verified)
        adm = [
            a for a in cands
            if ctx.accountant.admissible(profile.vector(a.action_class),
                                         reserve={} if a.mode.is_terminal else need)
        ]
        if not adm:
            adm = [Action(Mode.STOP_VERIFIED if ctx.verified else Mode.STOP_UNVERIFIED)]
        action, _ = arm.act(ctx, adm)
        realised = task.cost_of(action)
        _, trunc = _charge_or_truncate(ctx.accountant, action.action_class, realised)
        obs = task.step(action)
        _apply_observation(ctx, action, obs, channels, trunc)
        ctx.step += 1
    return task.succeeded()


def collect_branch_points(
    *,
    family: Family,
    split: str,
    seeds: range,
    profile: CostProfile,
    channels: dict[Tier, Channel],
    envelope: Envelope,
    fork_fractions: tuple[float, ...] = (0.3, 0.7),
    n_replicates: int = 24,
    tiers: tuple[Tier, ...] = tuple(Tier),
    max_steps: int = 40,
) -> list[BranchPoint]:
    """Run base episodes, fork at the requested points, try every candidate.

    `n_replicates` matters more than it looks. The outcome is binary and noisy, so
    a single rollout per (state, action) estimates a Bernoulli rate from one draw.
    With 24 replicates the standard error on a rate near 0.5 is about 0.10, which
    is the resolution limit on any ranking claim made here.
    """
    out: list[BranchPoint] = []

    for seed in seeds:
        # --- base trajectory, recording forkable snapshots -------------------
        base_task = family.task(seed)
        acc = Accountant(envelope=envelope)
        ctx = _fresh_context(base_task, acc, profile)
        arm = HeuristicArm()
        arm.reset(base_task)
        reserve = Reserve(profile)
        snapshots: list[tuple[int, SynthBug, EpisodeContext, list[Action]]] = []

        while not base_task.terminated and ctx.step < max_steps:
            cands = deterministic_candidates(ctx, tiers)
            need = reserve.required(verified=ctx.verified)
            adm = [
                a for a in cands
                if acc.admissible(profile.vector(a.action_class),
                                  reserve={} if a.mode.is_terminal else need)
            ]
            if not adm:
                break
            non_terminal = [a for a in adm if not a.mode.is_terminal]
            if len(non_terminal) >= 2:
                snapshots.append(
                    (ctx.step, copy.deepcopy(base_task), copy.deepcopy(ctx), list(non_terminal))
                )
            action, _ = arm.act(ctx, adm)
            realised = base_task.cost_of(action)
            _, trunc = _charge_or_truncate(acc, action.action_class, realised)
            obs = base_task.step(action)
            _apply_observation(ctx, action, obs, channels, trunc)
            ctx.step += 1

        if not snapshots:
            continue

        # --- fork at the requested fractions ---------------------------------
        for frac in fork_fractions:
            idx = min(len(snapshots) - 1, int(frac * len(snapshots)))
            step_i, snap_task, snap_ctx, candidates = snapshots[idx]

            # Cap the arm count so replicate budget is spent on precision rather
            # than on breadth; the deterministic generator orders by belief.
            candidates = candidates[:4]
            bp = BranchPoint(
                family=family.name, split=split, episode_seed=seed,
                decision_id=step_i, features=dict(snap_ctx.features()),
                n_replicates=n_replicates,
            )
            for cand in candidates:
                wins = 0
                for r in range(n_replicates):
                    t = copy.deepcopy(snap_task)
                    c = copy.deepcopy(snap_ctx)
                    c.accountant = copy.deepcopy(snap_ctx.accountant)
                    # Decorrelate replicate streams; without this every replicate
                    # replays identical randomness and the "rate" is one draw.
                    t._rng.seed(seed * 100_003 + step_i * 997 + r * 31 + hash(str(cand)) % 9973)
                    realised = t.cost_of(cand)
                    _, trunc = _charge_or_truncate(c.accountant, cand.action_class, realised)
                    obs = t.step(cand)
                    _apply_observation(c, cand, obs, channels, trunc)
                    c.step += 1
                    wins += int(_rollout(t, c, profile, channels, tiers, max_steps))
                bp.realised[str(cand)] = wins / n_replicates
            out.append(bp)
    return out


# -- metrics --------------------------------------------------------------------


def pairwise_ranking(
    points: list[BranchPoint], *, min_gap: float = 0.0
) -> dict[str, float]:
    """Fraction of within-state action pairs the model orders correctly.

    `min_gap` filters to pairs whose realised values differ by more than the given
    amount. Pairs that are genuinely tied carry no signal about ranking ability and
    only dilute the estimate toward 0.5.
    """
    correct = ties = total = 0
    for bp in points:
        acts = [a for a in bp.realised if a in bp.predicted]
        for i in range(len(acts)):
            for j in range(i + 1, len(acts)):
                a, b = acts[i], acts[j]
                dr = bp.realised[a] - bp.realised[b]
                if abs(dr) <= min_gap:
                    continue
                dp = bp.predicted[a] - bp.predicted[b]
                total += 1
                if dp == 0:
                    ties += 1
                elif (dp > 0) == (dr > 0):
                    correct += 1
    n = max(total, 1)
    return {
        "pairs": total,
        "accuracy": (correct + 0.5 * ties) / n,
        "ties": ties / n,
    }


def kendall_tau(points: list[BranchPoint]) -> float:
    """Mean within-state Kendall tau between predicted and realised values."""
    taus = []
    for bp in points:
        acts = [a for a in bp.realised if a in bp.predicted]
        if len(acts) < 2:
            continue
        conc = disc = 0
        for i in range(len(acts)):
            for j in range(i + 1, len(acts)):
                dp = bp.predicted[acts[i]] - bp.predicted[acts[j]]
                dr = bp.realised[acts[i]] - bp.realised[acts[j]]
                if dp == 0 or dr == 0:
                    continue
                if (dp > 0) == (dr > 0):
                    conc += 1
                else:
                    disc += 1
        if conc + disc:
            taus.append((conc - disc) / (conc + disc))
    return statistics.fmean(taus) if taus else 0.0


def decision_regret(points: list[BranchPoint]) -> dict[str, float]:
    """Realised value lost by choosing the model's argmax instead of the best."""
    regrets = []
    for bp in points:
        acts = [a for a in bp.realised if a in bp.predicted]
        if len(acts) < 2:
            continue
        chosen = max(acts, key=lambda a: bp.predicted[a])
        regrets.append(max(bp.realised[a] for a in acts) - bp.realised[chosen])
    if not regrets:
        return {}
    return {
        "n": len(regrets),
        "mean": statistics.fmean(regrets),
        "median": statistics.median(regrets),
        "frac_optimal": sum(1 for r in regrets if r <= 1e-9) / len(regrets),
    }


def oracle_spread(points: list[BranchPoint]) -> dict[str, float]:
    """How much the choice of action matters at all.

    If the best and worst action at a state produce the same outcome, no policy can
    beat any other and ranking accuracy is unmeasurable. This is the sanity check
    that the branch points are decision-relevant.
    """
    spreads = [max(bp.realised.values()) - min(bp.realised.values())
               for bp in points if len(bp.realised) >= 2]
    if not spreads:
        return {}
    return {
        "mean_spread": statistics.fmean(spreads),
        "median_spread": statistics.median(spreads),
        "frac_states_with_real_choice": sum(1 for s in spreads if s >= 0.10) / len(spreads),
    }
