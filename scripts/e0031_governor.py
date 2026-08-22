#!/usr/bin/env python3
"""E0031 -- does benchmark `difficulty` allocate compute? Preregistered.

Protocol in PREREGISTRATION-E0031-difficulty-governor.md, committed before any
fitting happened. Two phases, deliberately separate processes:

    --freeze     estimate three rescue rates and pick FRAC/K_EXTRA on
                 CALIBRATION only, write results/E0031_frozen.json, stop.
                 COMMIT THAT FILE.
    (default)    apply the frozen rule to the 225 evaluation problems, once.

Fitting and evaluating in one process makes `frozen_before_heldout`
unfalsifiable -- the same commit gets stamped as both -- so the evaluation
refuses to run until the freeze artifact exists and its recorded commit differs
from HEAD.

WHAT MAKES THIS DIFFERENT FROM E0029. `difficulty` is observable BEFORE the
first token is spent (E0030: CONTEST_RATING_DERIVED). Every earlier feature cost
a full generation to see. And `random` is scored alongside `best-fixed`, because
an allocator that beats uniform spend but not a random subset of equal size has
found the spending SHAPE, not a signal.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import pathlib
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from governor.harness.provenance import require_admissible
from governor.harness.traps import render, run_trap_checks

import scripts.e0029_analyse as A

ROOT = pathlib.Path(__file__).resolve().parents[1]
PROBLEMS = ROOT / "results" / "e0029_problems.json"
FROZEN = ROOT / "results" / "E0031_frozen.json"
RESULT = ROOT / "results" / "E0031_result.json"
LEVELS = ("easy", "medium", "hard")
N_BOOT = 4000


def _rel(p: pathlib.Path) -> str:
    """A diagnostic print must never be the thing that kills the run; fixture
    paths legitimately sit outside the repo."""
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def _head() -> str:
    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=str(ROOT),
                           capture_output=True, text=True)
        return r.stdout.strip() or "unknown"
    except Exception:                                     # noqa: BLE001
        return "unknown"


def tiebreak(qid: str) -> float:
    """Deterministic, arbitrary, independent of every outcome. Preregistered
    because a 3-level feature is mostly ties, and an unspecified ordering is one
    that can be chosen after seeing results."""
    return int(hashlib.sha256(f"E0031:{qid}".encode()).hexdigest()[:16], 16) / 2**64


def load(byq, qids, difficulty):
    """Per-problem arrays. `elig` marks problems whose sample 1 failed."""
    U1 = np.array([1.0 if byq[q][0]["hidden_all_passed"] else 0.0 for q in qids])
    Ua = np.array([1.0 if any(r["hidden_all_passed"] for r in byq[q]) else 0.0
                   for q in qids])
    diff = np.array([difficulty.get(q, "?") for q in qids])
    pub = np.array([float(byq[q][0].get("pub_frac", 0.0)) for q in qids])
    return U1, Ua, diff, pub


def rescue_rates(byq, qids, difficulty) -> dict:
    """v(d) = P(a later sample succeeds | difficulty d, sample 1 failed)."""
    num = collections.Counter()
    den = collections.Counter()
    for q in qids:
        s = byq[q]
        if s[0]["hidden_all_passed"]:
            continue
        d = difficulty.get(q, "?")
        den[d] += 1
        if any(r["hidden_all_passed"] for r in s[1:]):
            num[d] += 1
    return {d: {"rescued": num[d], "eligible": den[d],
                "rate": (num[d] / den[d]) if den[d] else 0.0} for d in LEVELS}


def make_arms(byq, qids, difficulty, v):
    """Score vectors for every arm. Higher score = spend sooner.

    Ineligible problems (sample 1 already solved) score -inf everywhere: there
    is nothing to buy, and letting them absorb budget would flatter every arm
    equally while measuring nothing.
    """
    U1, Ua, diff, pub = load(byq, qids, difficulty)
    rng = np.random.default_rng(20310)
    neg = -np.inf

    gov, rnd, myo, orc = [], [], [], []
    for i, q in enumerate(qids):
        if U1[i] > 0:
            gov.append(neg); rnd.append(neg); myo.append(neg); orc.append(neg)
            continue
        gov.append(v.get(diff[i], {}).get("rate", 0.0) + 1e-6 * tiebreak(q))
        rnd.append(rng.random())
        myo.append(pub[i] + 1e-9 * tiebreak(q))
        orc.append(Ua[i] + 1e-6 * tiebreak(q))
    return {"governor": np.array(gov), "random": np.array(rnd),
            "myopic": np.array(myo), "oracle": np.array(orc)}


def simulate(byq, qids, score, frac, k_extra):
    """Top `frac` of ELIGIBLE problems get k_extra more samples."""
    elig = [i for i, s in enumerate(score) if np.isfinite(s)]
    n_take = int(round(frac * len(elig)))
    take = set(np.array(elig)[np.argsort(-score[elig])[:n_take]]) if n_take else set()
    U, C = [], []
    for i, q in enumerate(qids):
        used = byq[q][:1 + k_extra] if i in take else byq[q][:1]
        U.append(1.0 if any(r["hidden_all_passed"] for r in used) else 0.0)
        C.append(sum(float(r.get("total_tokens", 0)) for r in used))
    return np.array(U), np.array(C)


def fixed_at(byq, qids, cost):
    """Best fixed k-for-everyone, interpolated to exactly match `cost`."""
    kmax = max(len(byq[q]) for q in qids)
    cs = [float(np.mean([sum(float(r.get("total_tokens", 0))
                             for r in byq[q][:k]) for q in qids]))
          for k in range(1, kmax + 1)]
    per = [np.array([1.0 if any(r["hidden_all_passed"] for r in byq[q][:k])
                     else 0.0 for q in qids]) for k in range(1, kmax + 1)]
    us = [p.mean() for p in per]
    if cost <= cs[0]:
        return us[0] * cost / cs[0], per[0] * cost / cs[0]
    for i in range(len(cs) - 1):
        if cs[i] <= cost <= cs[i + 1]:
            w = (cost - cs[i]) / (cs[i + 1] - cs[i])
            return us[i] + w * (us[i + 1] - us[i]), (1 - w) * per[i] + w * per[i + 1]
    return us[-1], per[-1]


def paired_ci(d, rng, n=N_BOOT):
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)]
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def load_everything():
    require_admissible(["difficulty"])                # aborts if reclassified
    byq, cal, ev, meta = A.load_joined()
    if (meta["cal"], meta["eval"]) != (A.EXPECTED_CAL, A.EXPECTED_EVAL):
        raise SystemExit(f"wrong dataset: split {meta['cal']}/{meta['eval']}")
    difficulty = {p["qid"]: p.get("difficulty", "?")
                  for p in json.loads(PROBLEMS.read_text())}
    missing = [q for q in cal + ev if q not in difficulty]
    if missing:
        raise SystemExit(f"{len(missing)} problems lack a difficulty label")
    return byq, cal, ev, meta, difficulty


def do_freeze(byq, cal, difficulty) -> dict:
    print("  === FREEZE (calibration only, n=%d) ===" % len(cal))
    v = rescue_rates(byq, cal, difficulty)
    print(f"    {'difficulty':10s} {'eligible':>9} {'rescued':>8} {'v(d)':>7}")
    for d in LEVELS:
        r = v[d]
        print(f"    {d:10s} {r['eligible']:9d} {r['rescued']:8d} {r['rate']:7.3f}")

    arms = make_arms(byq, cal, difficulty, v)
    kmax = max(len(byq[q]) for q in cal) - 1
    best = None
    for frac in np.linspace(0.1, 1.0, 10):
        for k in range(1, kmax + 1):
            U, C = simulate(byq, cal, arms["governor"], frac, k)
            ub, _ = fixed_at(byq, cal, float(C.mean()))
            adv = float(U.mean() - ub)
            if best is None or adv > best[0]:
                best = (adv, float(frac), int(k))
    adv, frac, k = best
    print(f"\n    frozen: FRAC={frac:.2f}  K_EXTRA={k}  "
          f"(calibration advantage {adv:+.4f})")
    frozen = {
        "commit": _head(), "frac": frac, "k_extra": k,
        "cal_advantage": adv, "v": v,
        "cal_ids_sha256": hashlib.sha256(json.dumps(sorted(cal)).encode()).hexdigest(),
        "n_cal": len(cal),
        "preregistration": "preregistrations/PREREGISTRATION-E0031-difficulty-governor.md",
    }
    FROZEN.write_text(json.dumps(frozen, indent=1))
    print(f"    wrote {_rel(FROZEN)} at commit {frozen['commit']}")
    print("\n    COMMIT THIS FILE, then run without --freeze.")
    return frozen


def do_evaluate(byq, cal, ev, difficulty) -> int:
    if not FROZEN.exists():
        raise SystemExit(f"no frozen rule at {_rel(FROZEN)}. "
                         "Run --freeze first, commit it, then evaluate.")
    fz = json.loads(FROZEN.read_text())
    if hashlib.sha256(json.dumps(sorted(cal)).encode()).hexdigest() != fz["cal_ids_sha256"]:
        raise SystemExit("the frozen rule was fitted on a different calibration set")

    frac, k, v = fz["frac"], fz["k_extra"], fz["v"]
    head = _head()
    print(f"  === EVALUATION (single pass, n={len(ev)}) ===")
    print(f"    frozen at {fz['commit']}, evaluating at {head}")
    if fz["commit"] == head:
        print("    WARNING: same commit -- the freeze was never committed, and")
        print("             frozen_before_heldout will correctly go RED.")
    print(f"    FRAC={frac:.2f}  K_EXTRA={k}")
    print(f"    v(easy)={v['easy']['rate']:.3f}  v(medium)={v['medium']['rate']:.3f}"
          f"  v(hard)={v['hard']['rate']:.3f}\n")

    arms = make_arms(byq, ev, difficulty, v)
    rng = np.random.default_rng(0)

    Ug, Cg = simulate(byq, ev, arms["governor"], frac, k)
    ub, Ub = fixed_at(byq, ev, float(Cg.mean()))

    print(f"    {'arm':12s} {'U':>8} {'cost':>10} {'vs fixed':>10} {'95% CI':>22}")
    rows = {}
    for name in ("governor", "random", "myopic", "oracle"):
        U, C = simulate(byq, ev, arms[name], frac, k)
        b, Bv = fixed_at(byq, ev, float(C.mean()))
        d = U - Bv
        lo, hi = paired_ci(d, rng)
        rows[name] = {"U": float(U.mean()), "cost": float(C.mean()),
                      "vs_fixed": float(U.mean() - b), "ci": [lo, hi],
                      "_U": U}
        print(f"    {name:12s} {U.mean():8.4f} {C.mean():10.1f} "
              f"{U.mean()-b:+10.4f}   [{lo:+.4f},{hi:+.4f}]")
    print(f"    {'best-fixed':12s} {ub:8.4f} {Cg.mean():10.1f}")

    # The comparison the preregistration says decides it.
    dgr = rows["governor"]["_U"] - rows["random"]["_U"]
    glo, ghi = paired_ci(dgr, rng)
    print(f"\n    governor - random: {dgr.mean():+.4f}  [{glo:+.4f}, {ghi:+.4f}]")

    beats_fixed = rows["governor"]["ci"][0] > 0
    beats_random = glo > 0
    ceiling = 0.1378
    print(f"    ceiling {ceiling:+.4f}   captured "
          f"{rows['governor']['vs_fixed']/ceiling:.1%}")

    evd = {
        "gov_utils": rows["governor"]["_U"], "greedy_utils": Ub,
        "gov_calls": np.isfinite(arms["governor"]).astype(float),
        "greedy_calls": np.full(len(ev), frac),
        "decisions_by_state": [(float(v.get(difficulty[q], {}).get("rate", 0.0)),)
                               for q in ev],
        "feature_names": ["difficulty"],
        "answered_rate": 1.0, "utility": rows["governor"]["U"],
        "requested": Cg, "actual_used": Cg, "charged": Cg,
        "scored_via_executor": True,
        "decisions": [int(np.isfinite(arms["governor"][i])) for i in range(len(ev))],
        "cell_ids": [difficulty[q] for q in ev],
        "froze_commit": fz["commit"], "heldout_commit": head,
        "selection_item_ids": cal, "evaluation_item_ids": ev,
        "token_cost_source": "exact tokenizer count over locally generated Qwen samples",
        "realised_cost": float(Cg.mean()), "budget": float(Cg.mean()),
        "baseline_cost": float(Cg.mean()),
        "cited_experiment_ids": ["E0029-QWEN-corrected", "E0030-difficulty-provenance"],
        "withdrawn_ids": ["E0019-predictor-loss-math", "E0017-soft-governor-math",
                          "E0029-QWEN-original"],
    }
    traps = run_trap_checks(evd)
    print("\n" + render(traps))
    failed = [k_ for k_, (ok, _) in traps.items() if not ok]

    verdict = ("BLOCKED" if failed else
               "PASS" if (beats_fixed and beats_random) else
               "NEGATIVE")
    print(f"\n  VERDICT: {verdict}")
    print(f"    beats best-fixed : {beats_fixed}")
    print(f"    beats random     : {beats_random}   <- both required")
    if failed:
        print(f"    blocked by: {failed}")

    RESULT.write_text(json.dumps({
        "experiment_id": "E0031-difficulty-governor", "verdict": verdict,
        "frozen": {k_: v_ for k_, v_ in fz.items()},
        "n_eval": len(ev), "ceiling": ceiling,
        "best_fixed_U": float(ub),
        "arms": {n: {k_: v_ for k_, v_ in r.items() if not k_.startswith("_")}
                 for n, r in rows.items()},
        "governor_minus_random": {"mean": float(dgr.mean()), "ci": [glo, ghi]},
        "beats_fixed": bool(beats_fixed), "beats_random": bool(beats_random),
        "traps_failed": failed,
    }, indent=1))
    print(f"    wrote {_rel(RESULT)}")
    return 0 if verdict == "PASS" else 2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--freeze", action="store_true")
    args = ap.parse_args()
    print("\nE0031 -- difficulty-based allocation, preregistered\n")
    byq, cal, ev, meta, difficulty = load_everything()
    print(f"  problems {meta['problems']}  cal {meta['cal']}  eval {meta['eval']}")
    print(f"  difficulty admissible: E0030 CONTEST_RATING_DERIVED\n")
    if args.freeze:
        do_freeze(byq, cal, difficulty)
        return 0
    return do_evaluate(byq, cal, ev, difficulty)


if __name__ == "__main__":
    raise SystemExit(main())
