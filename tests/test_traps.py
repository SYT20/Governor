"""Each trap check must FIRE on the historical case that motivated it."""
from __future__ import annotations
import numpy as np
from governor.harness import traps as T


def test_greedy_collapse_fires_on_env5_signature():
    u = [0.5, 0.7, 0.3]
    assert not T.greedy_collapse(u, u, [2, 2, 2], [2, 2, 2])[0]
    assert T.greedy_collapse([0.6, 0.7, 0.4], u, [2, 2, 2], [2, 2, 2])[0]


def test_oracle_leakage_fires_on_sigma_other():
    assert not T.oracle_leakage(["cue", "sigma_other"])[0]
    assert T.oracle_leakage(["cue", "max_py"])[0]


def test_answered_vs_utility_fires_on_429_signature():
    assert not T.answered_vs_utility(0.16, 0.16)[0]      # Gemini throttling
    assert T.answered_vs_utility(1.0, 0.80)[0]


def test_token_accounting_rejects_nominal_cost():
    assert not T.token_accounting([500], [312], [1.0])[0]   # charged nominal
    assert T.token_accounting([500], [312], [312])[0]


def test_progress_as_cognition_fires():
    assert not T.progress_as_cognition(["H_y", "n_blocks_touched"])[0]
    assert T.progress_as_cognition(["H_y", "n_blocks_touched"],
                                   justified={"n_blocks_touched"})[0]


def test_invariant_as_intelligence_fires_on_lookup():
    assert not T.invariant_as_intelligence([1, 1, 0, 0], ["a", "a", "b", "b"])[0]
    assert T.invariant_as_intelligence([1, 0, 0, 0], ["a", "a", "b", "b"])[0]


def test_mc_convergence_fires_when_sd_collapses():
    e = {32: np.array([0.2, -0.2, 0.1]), 512: np.array([0.01, -0.01, 0.005])}
    assert not T.mc_convergence(e)[0]


def test_missing_evidence_is_RED_not_silent():
    """Silence must not read as success. Every check REQUIRING evidence goes
    red when it is absent. secret_scan is exempt: it needs no evidence and
    legitimately passes on a clean tree."""
    r = T.run_trap_checks({})
    needs_evidence = {k: v for k, v in r.items() if k != "secret_scan"}
    assert all(not ok for ok, _ in needs_evidence.values())
    assert all("missing evidence" in d for _, d in needs_evidence.values())
    assert r["secret_scan"][0], "clean tree should pass the secret scan"


# -- Phase 4R: selection/evaluation split ---------------------------------------

def test_split_leakage_fires_on_any_overlap():
    """The regression test the reviewer asked for: an experiment that selects a
    configuration from one item set must fail if an evaluation item appears in
    that set."""
    from governor.harness.traps import split_leakage
    ok, _ = split_leakage(["a", "b", "c"], ["d", "e"])
    assert ok
    ok, detail = split_leakage(["a", "b", "c"], ["c", "d"])
    assert not ok and "overlap=1" in detail


def test_frozen_split_is_disjoint_and_stable():
    from governor.phase4.split import build, evaluation_ids, selection_ids, verify_disjoint
    ok, detail = verify_disjoint()
    assert ok, detail
    assert not (selection_ids() & evaluation_ids())
    assert build()["sha256"] == build()["sha256"]


def test_split_is_independent_of_cache_state():
    """Freezing by item id, not by list slice: the split must not move as
    collection proceeds."""
    from governor.phase4.split import build
    a = build()
    assert len(a["selection_ids"]) == 40
    assert len(set(a["selection_ids"])) == 40
    assert a["selection_ids"] == build()["selection_ids"]


def test_run_trap_checks_reds_missing_split_evidence():
    from governor.harness.traps import run_trap_checks
    out = run_trap_checks({})
    assert out["split_leakage"][0] is False
    assert "NOT RUN" in out["split_leakage"][1]
