"""Executable trap checks. HARD GATES — a red check blocks a scientific PASS.

Every trap here was caught by a human noticing something odd in output that had
ALREADY PRINTED "PASS". Three of them today. A warning next to a green verdict
gets skimmed; these return a verdict that callers must respect.

Each check takes evidence and returns (ok, detail). Callers do:

    red = [n for n, (ok, _) in run_trap_checks(ev).items() if not ok]
    if red: verdict = "BLOCKED"
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

FORBIDDEN_FEATURES = {
    "hidden_difficulty", "difficulty", "hard", "regime", "sigma_other",
    "config_id", "configuration", "episode_id", "seed", "oracle", "oracle_delta",
    "future", "label", "answer", "ground_truth", "p_hard", "cue_noise",
}
PROGRESS_FEATURES = {"t", "step", "n_calls", "n_acquired", "n_blocks_touched",
                     "progress", "position"}


def greedy_collapse(gov_utils, greedy_utils, gov_calls, greedy_calls, tol=1e-12):
    """Env 5: a Governor scored Δ=+0.0000 with identical calls/episode. It had
    collapsed into greedy and the threshold never discriminated."""
    d = np.asarray(gov_utils) - np.asarray(greedy_utils)
    same_calls = np.allclose(gov_calls, greedy_calls, atol=tol)
    identical = bool(np.all(np.abs(d) <= tol)) and same_calls
    return (not identical,
            f"mean_delta={d.mean():+.6f} calls_identical={same_calls}")


def constant_schedule(decisions_by_state):
    """Env 5: 'never spend early' beat everything. If allocation is constant
    across observable states it is a schedule, not a controller."""
    uniq = {tuple(v) for v in decisions_by_state}
    return len(uniq) > 1, f"distinct_decision_patterns={len(uniq)}"


def oracle_leakage(feature_names):
    """Env 4a: the scorer read cfg.sigma_other while the docstring claimed it
    could not. Names are checked, not intentions."""
    bad = [f for f in feature_names
           if any(k in f.lower() for k in FORBIDDEN_FEATURES)]
    return not bad, f"forbidden_features={bad}"


def answered_vs_utility(answered_rate, utility, tol=1e-9):
    """Gemini: answered==utility exactly meant 492/500 requests were 429s.
    Every call returning anything was correct; the rest returned nothing."""
    coincide = abs(answered_rate - utility) <= tol and answered_rate < 0.999
    return (not coincide,
            f"answered={answered_rate:.4f} utility={utility:.4f} "
            f"-> {'INSPECT PROVIDER ERRORS' if coincide else 'ok'}")


def token_accounting(requested, actual_used, charged, tol=1e-6):
    """Never charge nominal cost. MathM2 charged 1.0; an LLM must charge what
    the runtime reports."""
    ok = np.allclose(actual_used, charged, atol=tol)
    over = np.asarray(actual_used) > np.asarray(requested) + tol
    return (ok and not over.any(),
            f"charged==used:{ok} over_budget_calls={int(over.sum())}")


def execution_vs_scoring(scored_via_executor: bool):
    """Run I: scoring and execution were separate code paths, so no policy was
    ever executed. This must be asserted by the caller, structurally."""
    return scored_via_executor, f"scored_through_canonical_executor={scored_via_executor}"


def progress_as_cognition(feature_names, justified=frozenset()):
    """Env 5: n_blocks_touched was filed as cognitive state; moving it into the
    progress control killed a +0.035 'cognitive' residual."""
    prog = [f for f in feature_names
            if f.lower() in PROGRESS_FEATURES and f not in justified]
    return not prog, f"unjustified_progress_features={prog}"


def mc_convergence(estimates_by_k: dict[int, np.ndarray]):
    """CUBE-NM: n_mc=32 inverted the conclusion. If between-state variance
    keeps shrinking as k grows, the 'signal' is unremoved estimator noise."""
    ks = sorted(estimates_by_k)
    if len(ks) < 2:
        return False, "need >=2 sample sizes"
    sds = [float(np.std(estimates_by_k[k], ddof=1)) for k in ks]
    stable = sds[-1] >= 0.5 * sds[-2]
    return stable, f"sd_by_k={dict(zip(ks, [round(s, 4) for s in sds]))}"


def invariant_as_intelligence(decisions, cell_ids):
    """The any-time switch: escalation was 0% or 100% in 24/24 cells and
    constant within every (cost,budget) cell — a 6-entry lookup."""
    cells = {}
    for d, c in zip(decisions, cell_ids):
        cells.setdefault(c, set()).add(d)
    constant = [c for c, v in cells.items() if len(v) == 1]
    ok = len(constant) < len(cells)
    return ok, f"cells_with_constant_decision={len(constant)}/{len(cells)}"


def frozen_before_heldout(froze_commit: str, heldout_commit: str):
    """Env 5: sigma=0.35/B=4 was selected BECAUSE its aggregate sat near zero,
    then measured on the same states."""
    ok = bool(froze_commit) and froze_commit != heldout_commit
    return ok, f"froze={froze_commit[:8] or 'MISSING'} heldout={heldout_commit[:8]}"


def split_leakage(selection_item_ids, evaluation_item_ids):
    """Phase 4R: an experiment that CHOOSES a configuration from one item set
    must not then EVALUATE it on items from that same set.

    A reviewer suspected exactly this in the Phase 4R structural search. The
    record showed the winning configuration had in fact qualified on 40 items
    while only rejected alternatives saw a larger pool -- but establishing that
    required reading a metrics file, which is not a protocol. Overlap is now a
    red check, so no future run can be ambiguous about it.
    """
    s, e = set(selection_item_ids), set(evaluation_item_ids)
    both = s & e
    return (not both,
            f"selection={len(s)} evaluation={len(e)} overlap={len(both)}"
            + (f" e.g. {sorted(both)[:3]}" if both else ""))


def withdrawn_result_promotion(cited_experiment_ids, withdrawn_ids):
    """A withdrawn result must never reappear as current evidence.

    The ledger is append-only, so `E0019-predictor-loss-math` still reads PASS
    on disk even though its Governor overspent by 15% and the corrected E0021
    superseded it. Preserving the row is right; citing it is not. This check
    separates the two.
    """
    cited, gone = set(cited_experiment_ids), set(withdrawn_ids)
    bad = sorted(cited & gone)
    return (not bad,
            f"cited={len(cited)} withdrawn={len(gone)} promoted={bad}")


def budget_adherence(realised_cost, budget, baseline_cost=None, tol=0.02):
    """A policy compared at a budget it did not respect is not a comparison.

    E0019 reported the Governor beating the best fixed policy by +0.0282 "at
    identical expected budget". It was spending 973 tokens against a budget of
    846 -- 15% over -- because the Lagrangian was tuned to hit the budget on the
    CALIBRATION half and the evaluation half costs more. Scored against a fixed
    baseline given the same 973 tokens, the Governor LOST by 0.0131.

    Tuning on calibration is correct; reporting the baseline at the nominal
    budget rather than at the realised cost is not. Always compare at matched
    REALISED cost.
    """
    over = float(realised_cost) / float(budget) - 1.0 if budget else float("inf")
    ok = over <= tol                       # UNDER-spending is allowed
    detail = f"realised={realised_cost:.0f} budget={budget:.0f} over={over:+.1%}"
    if baseline_cost is not None:
        # The baseline must not have been given LESS resource than the policy.
        # A baseline that spent MORE is fine -- that handicaps the policy.
        short = 1.0 - float(baseline_cost) / float(realised_cost)
        ok = ok and short <= tol
        detail += (f" baseline={baseline_cost:.0f} "
                   f"baseline_short_by={short:+.1%}")
    return ok, detail


def exact_token_counts(source: str, tolerance: float = 0.05):
    """Token costs must come from a TOKENIZER, never an estimate.

    E0013 costed generations at len(text)/4. Against the real
    `simplescaling/s1-32B` tokenizer that was off by 25-35%, AND the error moved
    with the budget level (0.651 at one Wait, 0.756 at eight), so it was not even
    a consistent rescaling -- it distorted the very ratios the allocation
    analysis depends on. An estimated resource unit measures a simulated
    resource.
    """
    ok = isinstance(source, str) and source.strip().lower() not in (
        "", "estimate", "approx", "len/4", "heuristic", "chars/4")
    return ok, f"token_cost_source={source!r}"


def secret_scan(root="."):
    pats = [re.compile(p) for p in (r"sk-" + r"or-v1-[A-Za-z0-9]{16,}",
                                    r"sk-" + r"ant-[A-Za-z0-9\-]{16,}",
                                    r"AIza[A-Za-z0-9_\-]{30,}")]
    hits = []
    for f in Path(root).rglob("*"):
        if not f.is_file() or ".git" in f.parts or "traps.py" in f.name:
            continue
        if f.suffix not in {".py", ".md", ".json", ".sh", ".txt"}:
            continue
        try:
            t = f.read_text(errors="ignore")
        except OSError:
            continue
        if any(p.search(t) for p in pats):
            hits.append(str(f))
    return not hits, f"files_with_credentials={hits}"


def run_trap_checks(ev: dict) -> dict[str, tuple[bool, str]]:
    """Run every check for which evidence was supplied. MISSING EVIDENCE IS A
    RED CHECK -- silence must not read as success."""
    out: dict[str, tuple[bool, str]] = {}
    reg = {
        "greedy_collapse": ("gov_utils", "greedy_utils", "gov_calls", "greedy_calls"),
        "constant_schedule": ("decisions_by_state",),
        "oracle_leakage": ("feature_names",),
        "answered_vs_utility": ("answered_rate", "utility"),
        "token_accounting": ("requested", "actual_used", "charged"),
        "execution_vs_scoring": ("scored_via_executor",),
        "progress_as_cognition": ("feature_names",),
        "invariant_as_intelligence": ("decisions", "cell_ids"),
        "frozen_before_heldout": ("froze_commit", "heldout_commit"),
        "split_leakage": ("selection_item_ids", "evaluation_item_ids"),
        "exact_token_counts": ("token_cost_source",),
        "budget_adherence": ("realised_cost", "budget"),
        "withdrawn_result_promotion": ("cited_experiment_ids",
                                       "withdrawn_ids"),
        "secret_scan": (),
    }
    fns = globals()
    for name, args in reg.items():
        if all(a in ev for a in args):
            out[name] = fns[name](*[ev[a] for a in args])
        else:
            missing = [a for a in args if a not in ev]
            out[name] = (False, f"NOT RUN - missing evidence: {missing}")
    if "estimates_by_k" in ev:
        out["mc_convergence"] = mc_convergence(ev["estimates_by_k"])
    return out


def render(results: dict[str, tuple[bool, str]]) -> str:
    lines = []
    for n, (ok, d) in results.items():
        lines.append(f"  {'GREEN' if ok else '  RED'}  {n:<26} {d}")
    red = [n for n, (ok, _) in results.items() if not ok]
    lines.append(f"\n  {len(red)} RED -> " +
                 ("PASS BLOCKED: " + ", ".join(red) if red else "no traps triggered"))
    return "\n".join(lines)
