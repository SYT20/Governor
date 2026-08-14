"""Corpus sufficiency gate — Decision Record section J.8.

Revision 1 gated on "at least 30 observations per action class". That gate is close
to meaningless, and the external review was right to flag it: checkpoints drawn from
the same episode are not independent observations. Counting them as though they were
lets a corpus declare itself ready on the strength of correlated data.

This module replaces the raw count with a clustering-adjusted effective sample size,
and computing it properly turns up something worth stating plainly:

    Every checkpoint in an episode carries the *same* label -- the eventual outcome
    of that episode. A variable that is constant within a cluster has an
    intra-class correlation of exactly 1, so the design effect is the full cluster
    size and the effective sample size collapses to the number of *episodes*.

Ten thousand checkpoints drawn from 200 episodes carry, for the purpose of
predicting episode outcome, the information of 200 observations. Not 10,000. The
gate is therefore stricter than revision 1 implied, and the fix when it fails is
more *diverse episodes*, never more checkpoints from the ones you already have.

(Within-episode feature variation does carry information about *which states*
precede success, so the picture is not quite as bleak as ESS alone suggests. But
ESS is the conservative number, and a gate should be conservative.)
"""

from __future__ import annotations

import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass

from governor.corpus.build import Checkpoint

# Thresholds (section J.8).
MIN_ESS = 200
MIN_EPISODES_PER_CELL = 25
MIN_PER_OUTCOME = 40
MIN_EPISODES_PER_FAMILY = 20
MIN_FAMILIES = 8


def intra_class_correlation(values: list[float], groups: list[str]) -> float:
    """One-way random-effects ICC.

    Returns 1.0 for a variable that is constant within every group, which is
    exactly the case for episode-outcome labels.
    """
    by: dict[str, list[float]] = defaultdict(list)
    for v, g in zip(values, groups):
        by[g].append(v)
    by = {g: vs for g, vs in by.items() if vs}
    k = len(by)
    n = sum(len(vs) for vs in by.values())
    if k < 2 or n <= k:
        return 0.0

    grand = sum(sum(vs) for vs in by.values()) / n
    ss_between = sum(len(vs) * (statistics.fmean(vs) - grand) ** 2 for vs in by.values())
    ss_within = sum(
        sum((v - statistics.fmean(vs)) ** 2 for v in vs) for vs in by.values()
    )
    ms_between = ss_between / (k - 1)
    ms_within = ss_within / (n - k)
    m_bar = n / k

    if ms_between <= 0:
        return 0.0
    if ms_within == 0:
        return 1.0  # constant within clusters
    icc = (ms_between - ms_within) / (ms_between + (m_bar - 1) * ms_within)
    return max(0.0, min(1.0, icc))


def effective_sample_size(values: list[float], groups: list[str]) -> tuple[float, float, float]:
    """Return (ess, icc, design_effect) for a clustered sample."""
    n = len(values)
    k = len(set(groups))
    if n == 0 or k == 0:
        return 0.0, 0.0, 1.0
    m_bar = n / k
    icc = intra_class_correlation(values, groups)
    deff = 1.0 + (m_bar - 1.0) * icc
    return n / deff, icc, deff


@dataclass(slots=True)
class GateResult:
    passed: bool
    checks: list[dict[str, object]]
    stats: dict[str, object]

    def render(self) -> str:
        lines = []
        for c in self.checks:
            mark = "PASS" if c["ok"] else "FAIL"
            lines.append(f"    {c['name']:<44} {c['observed']:>18}  need {c['required']:<10} {mark}")
        return "\n".join(lines)


def check_gate(checkpoints: list[Checkpoint], *, split: str = "train") -> GateResult:
    """Evaluate every J.8 condition against the collected corpus."""
    ck = [c for c in checkpoints if c.split == split]
    checks: list[dict[str, object]] = []

    if not ck:
        return GateResult(False, [{"name": "corpus non-empty", "observed": 0,
                                   "required": ">0", "ok": False}], {})

    labels = [float(c.label) for c in ck]
    episodes = [c.episode_id for c in ck]
    ess, icc, deff = effective_sample_size(labels, episodes)

    n_ck = len(ck)
    n_ep = len(set(episodes))
    ep_label = {c.episode_id: c.label for c in ck}
    n_pos = sum(1 for v in ep_label.values() if v == 1)
    n_neg = sum(1 for v in ep_label.values() if v == 0)

    # 1. Effective sample size.
    checks.append({"name": "effective sample size (label)", "observed": f"{ess:.0f}",
                   "required": f">={MIN_ESS}", "ok": ess >= MIN_ESS})

    # 2. Distinct episodes per (mode, tier) cell -- episodes, not checkpoints.
    cell_eps: dict[tuple[str, str], set[str]] = defaultdict(set)
    for c in ck:
        if not c.mode.startswith("STOP"):
            cell_eps[(c.mode, c.tier)].add(c.episode_id)
    worst_cell, worst_n = None, 10**9
    for cell, eps in cell_eps.items():
        if len(eps) < worst_n:
            worst_cell, worst_n = cell, len(eps)
    if worst_cell is None:
        worst_cell, worst_n = ("-", "-"), 0
    checks.append({"name": f"episodes per (mode,tier) cell [worst: {worst_cell[0]}@{worst_cell[1]}]",
                   "observed": worst_n, "required": f">={MIN_EPISODES_PER_CELL}",
                   "ok": worst_n >= MIN_EPISODES_PER_CELL})

    # 3. Both outcome classes present in usable quantity.
    checks.append({"name": "successful episodes", "observed": n_pos,
                   "required": f">={MIN_PER_OUTCOME}", "ok": n_pos >= MIN_PER_OUTCOME})
    checks.append({"name": "failed episodes", "observed": n_neg,
                   "required": f">={MIN_PER_OUTCOME}", "ok": n_neg >= MIN_PER_OUTCOME})

    # 4. Family coverage -- the stand-in for repo diversity.
    fam_eps: dict[str, set[str]] = defaultdict(set)
    for c in ck:
        fam_eps[c.family].add(c.episode_id)
    n_fam = len(fam_eps)
    min_fam = min((len(v) for v in fam_eps.values()), default=0)
    checks.append({"name": "distinct families", "observed": n_fam,
                   "required": f">={MIN_FAMILIES}", "ok": n_fam >= MIN_FAMILIES})
    checks.append({"name": "episodes per family (min)", "observed": min_fam,
                   "required": f">={MIN_EPISODES_PER_FAMILY}", "ok": min_fam >= MIN_EPISODES_PER_FAMILY})

    # 5. Drift check #4: randomised action coverage. Without a randomised
    #    observation in a cell, that cell's effect is confounded by the behaviour
    #    policy and D1 cannot be settled there.
    rand_cells = {(c.mode, c.tier) for c in ck if c.was_random and not c.mode.startswith("STOP")}
    all_cells = set(cell_eps)
    uncovered = sorted(all_cells - rand_cells)
    checks.append({"name": "cells with randomised coverage",
                   "observed": f"{len(rand_cells)}/{len(all_cells)}",
                   "required": "all", "ok": not uncovered})

    stats = {
        "checkpoints": n_ck,
        "episodes": n_ep,
        "mean_checkpoints_per_episode": round(n_ck / max(n_ep, 1), 2),
        "icc_label": round(icc, 4),
        "design_effect": round(deff, 2),
        "ess": round(ess, 1),
        "success_rate": round(n_pos / max(n_ep, 1), 4),
        "randomised_fraction": round(sum(c.was_random for c in ck) / max(n_ck, 1), 4),
        "uncovered_cells": uncovered,
        "action_histogram": dict(Counter(f"{c.mode}@{c.tier}" for c in ck).most_common()),
    }
    return GateResult(all(c["ok"] for c in checks), checks, stats)
