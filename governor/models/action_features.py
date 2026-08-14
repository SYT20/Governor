"""Action-conditioned features — the fix Stage 4B's negative result points to.

Stage 4A: pooled Q ranks states (AUC 0.74) far better than actions at a fixed
state (0.60). Stage 4B: an explicit V + advantage decomposition changed nothing
(0.599 -> 0.590, well inside noise). Together those say the limitation is not the
estimator or the factorisation of the target -- it is that the model has no
feature describing what a *particular* action would do in *this* state.

Every feature the model previously saw described the state. The action entered
only as a one-hot, so the best it could learn was a constant offset per action
class plus a budget interaction. Asked whether EXPLORE@T2->h2 beats
EXPLOIT@T1->h0 at a given belief, it had nothing to answer with.

The features below close that gap, and every one is deterministic arithmetic over
the belief vector and the *measured* channel reliability -- no model invents any
of them, which keeps the section D provenance invariant intact.

Note where expected information gain lands. Decision Record section 14 argued it
should stay implicit rather than becoming a hand-weighted additive utility term,
and that still holds: it is not added to EU with a lambda. It appears here as a
FEATURE, computed in closed form from the belief and the calibrated channel, and
the model decides what it is worth.
"""

from __future__ import annotations

import math

from governor.cognitive.belief import Channel, entropy

ACTION_FEATURE_NAMES = (
    "belief_in_target",
    "expected_info_gain",
    "tier_index",
    "channel_lr_plus",
    "is_explore",
    "is_exploit",
    "is_verify",
    "has_unverified_patch",
    "target_is_argmax",
)

_TIER_IDX = {"T0": 0.0, "T1": 1.0, "T2": 2.0}


def expected_info_gain(belief: list[float], target: int, ch: Channel) -> float:
    """Closed-form expected entropy reduction from probing `target`.

        EIG = H(pi) - E_o[ H(pi | o) ]

    with the observation distribution and both posteriors following directly from
    the channel's two measured parameters. This is the quantity section 14 wanted
    and refused to let an LLM invent; here it is exact arithmetic.
    """
    if not belief or not 0 <= target < len(belief):
        return 0.0
    pt = belief[target]
    p_one = pt * ch.alpha + (1.0 - pt) * ch.beta
    p_zero = 1.0 - p_one
    if p_one <= 1e-12 or p_zero <= 1e-12:
        return 0.0

    def posterior(obs: int) -> list[float]:
        lik = ch.likelihoods(len(belief), target, obs)
        w = [b * l for b, l in zip(belief, lik)]
        tot = sum(w)
        return [x / tot for x in w] if tot > 1e-12 else list(belief)

    h0 = entropy(belief)
    return h0 - (p_one * entropy(posterior(1)) + p_zero * entropy(posterior(0)))


def action_features(
    *,
    belief: list[float],
    mode: str,
    tier: str,
    target: int | None,
    channels: dict[str, Channel],
    n_exploit: float,
    n_verify: float,
) -> dict[str, float]:
    """Everything the policy can know about this action in this state."""
    ch = channels.get(tier)
    tgt_belief = belief[target] if (target is not None and 0 <= target < len(belief)) else 0.0
    eig = (
        expected_info_gain(belief, target, ch)
        if (mode == "EXPLORE" and target is not None and ch is not None)
        else 0.0
    )
    argmax = belief.index(max(belief)) if belief else -1
    return {
        "belief_in_target": tgt_belief,
        "expected_info_gain": eig,
        "tier_index": _TIER_IDX.get(tier, 0.0),
        "channel_lr_plus": math.log(ch.lr_positive) if ch is not None else 0.0,
        "is_explore": 1.0 if mode == "EXPLORE" else 0.0,
        "is_exploit": 1.0 if mode == "EXPLOIT" else 0.0,
        "is_verify": 1.0 if mode == "VERIFY" else 0.0,
        "has_unverified_patch": 1.0 if n_exploit > n_verify else 0.0,
        "target_is_argmax": 1.0 if (target is not None and target == argmax) else 0.0,
    }
