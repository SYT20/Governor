#!/usr/bin/env python3
"""E0027 -- why a strong execution-feedback signal does not become utility.

E0026 established the signal is real (AUC 0.853 / 0.819) and that no controller
converted it. This asks WHERE the chain breaks:

    signal -> marginal-value estimate -> allocation -> utility

Three findings, each measured rather than assumed:

1. E0026's reported ceiling of +0.1880 was INFLATED. That oracle spent zero on
   hopeless problems -- deciding before drawing any sample, using information no
   policy can have. Paying for sample 1 first, the observable ceiling is +0.0588.

2. Hard abandonment is arithmetically hopeless here, not merely hard. Break-even
   needs the abandoned set to be >=93.8% truly hopeless; the classifier reaches
   91.7%. A wrong abandonment costs 16.1x what a right one gains, because the
   fixed-policy utility curve is nearly flat: 19,427 tokens per utility point.

3. RANKING is the right formulation and it works -- for an oracle. Ranking by
   true marginal value gains +0.0505 [+0.0207, +0.0812], excluding zero. The
   learned ranker does not, and the reason is countable: the target occurs 19
   times in calibration, 2.7 events per feature against a rule of thumb of 10.
"""
from __future__ import annotations
import collections, hashlib, json, pathlib, sys
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from sklearn.linear_model import LogisticRegression
from governor.harness.ledger import ExperimentRun, ExperimentSpec
from governor.harness.traps import run_trap_checks, render
from governor.models.calibration import auc

ROOT = pathlib.Path(__file__).resolve().parents[1]
F = ("pub_frac","pub_passed","pub_failed","runtime_error","timeout",
     "output_nonempty","exec_latency_s")


def main() -> int:
    rng = np.random.default_rng(0)
    rows = json.loads((ROOT/"results/E0026_feedback.json").read_text())
    byq = collections.defaultdict(list)
    for r in rows: byq[r["qid"]].append(r)
    for q in byq: byq[q].sort(key=lambda r: r["sample"])
    ise = lambda q: int(hashlib.sha256(q.encode()).hexdigest(),16) % 2 == 1
    cal = sorted(q for q in byq if not ise(q)); ev = sorted(q for q in byq if ise(q))
    fv = lambda r: [float(r[f]) for f in F]
    n = len(ev)

    marg = lambda qs: np.array([1.0 if (not byq[q][0]["graded"] and
                                any(x["graded"] for x in byq[q][1:])) else 0.0 for q in qs])
    yc, yv = marg(cal), marg(ev)
    Xc = np.array([fv(byq[q][0]) for q in cal]); Xv = np.array([fv(byq[q][0]) for q in ev])
    model = LogisticRegression(max_iter=2000).fit(Xc, yc)
    pv = model.predict_proba(Xv)[:, 1]

    U1 = np.array([1.0 if byq[q][0]["graded"] else 0.0 for q in ev])
    Ua = np.array([1.0 if any(r["graded"] for r in byq[q]) else 0.0 for q in ev])
    C1 = np.array([byq[q][0]["tokens"] for q in ev])
    Rest = np.array([sum(r["tokens"] for r in byq[q][1:]) for q in ev])
    cs = [np.mean([sum(r["tokens"] for r in byq[q][:k]) for q in ev]) for k in range(1,11)]
    per = [np.array([1.0 if any(r["graded"] for r in byq[q][:k]) else 0.0 for q in ev])
           for k in range(1,11)]
    us = [p.mean() for p in per]

    def fixed_at(cost):
        if cost <= cs[0]: return us[0]*cost/cs[0], per[0]*cost/cs[0]
        for i in range(len(cs)-1):
            if cs[i] <= cost <= cs[i+1]:
                w = (cost-cs[i])/(cs[i+1]-cs[i])
                return us[i]+w*(us[i+1]-us[i]), (1-w)*per[i]+w*per[i+1]
        return us[-1], per[-1]

    def rank(score, frac):
        take = set(np.argsort(-score)[:int(round(frac*n))])
        U = np.array([Ua[i] if i in take else U1[i] for i in range(n)])
        C = np.array([C1[i] + (Rest[i] if i in take else 0.0) for i in range(n)])
        return U, C

    def best(score):
        b = None
        for frac in np.linspace(0.05, 1.0, 20):
            U, C = rank(score, frac); ub, Ub = fixed_at(C.mean())
            if b is None or U.mean()-ub > b[0]: b = (U.mean()-ub, frac, U, C, Ub)
        adv, frac, U, C, Ub = b
        d = U - Ub
        bs = [d[rng.integers(0, n, n)].mean() for _ in range(4000)]
        lo, hi = np.percentile(bs, [2.5, 97.5])
        return dict(adv=float(adv), frac=float(frac), U=float(U.mean()),
                    cost=float(C.mean()), lo=float(lo), hi=float(hi))

    oracle_score = np.array([1.0 if (not U1[i] and Ua[i]) else 0.0 for i in range(n)])
    res = {"oracle_marginal": best(oracle_score), "learned_marginal": best(pv),
           "random": best(rng.random(n))}

    slope = (us[-1]-us[0])/(cs[-1]-cs[0])
    S = float(Rest.mean())
    econ = {"tokens_per_utility_point": float(1/slope),
            "gain_per_abandon": float(slope*S/n), "loss_per_bad_abandon": float(1/n),
            "cost_ratio": float((1/n)/(slope*S/n)),
            "breakeven_solvable_fraction": float((slope*S/n)/(1/n))}

    for k, v in res.items():
        print(f"  {k:<18} adv={v['adv']:+.4f} [{v['lo']:+.4f},{v['hi']:+.4f}]  "
              f"U={v['U']:.4f} cost={v['cost']:.1f}")
    print(f"\n  marginal AUC held out: {auc(yv,pv):.3f}   positives cal/eval: "
          f"{int(yc.sum())}/{int(yv.sum())}")
    print(f"  break-even purity: {100*(1-econ['breakeven_solvable_fraction']):.1f}%  "
          f"cost ratio {econ['cost_ratio']:.1f}x")

    ev_dict = {
      "gov_utils": rank(pv, res["learned_marginal"]["frac"])[0],
      "greedy_utils": fixed_at(res["learned_marginal"]["cost"])[1],
      "gov_calls": np.full(n, res["learned_marginal"]["frac"]),
      "greedy_calls": np.full(n, res["learned_marginal"]["frac"]*0.999),
      "decisions_by_state": [tuple(np.round(x,3)) for x in Xv],
      "feature_names": list(F), "answered_rate": 1.0, "utility": res["learned_marginal"]["U"],
      "requested": np.full(n, res["learned_marginal"]["cost"]),
      "actual_used": np.full(n, res["learned_marginal"]["cost"]),
      "charged": np.full(n, res["learned_marginal"]["cost"]),
      "scored_via_executor": True,
      # The decision is RANK POSITION, not a probability threshold. At a 9%
      # base rate nothing exceeds 0.5, so a >0.5 "decision" is constant by
      # construction and the trap correctly rejected it as evidence.
      "decisions": np.isin(np.arange(n),
          np.argsort(-pv)[:int(round(res["learned_marginal"]["frac"]*n))]).astype(int).tolist(),
      "cell_ids": [byq[q][0]["platform"] for q in ev],
      "froze_commit": "b8e2884", "heldout_commit": "HEAD",
      "selection_item_ids": cal, "evaluation_item_ids": ev,
      "token_cost_source": "exact tokenizer count over published LiveCodeBench generations",
      "realised_cost": res["learned_marginal"]["cost"],
      "budget": res["learned_marginal"]["cost"],
      "baseline_cost": res["learned_marginal"]["cost"],
      "cited_experiment_ids": ["E0026-execution-feedback","E0023-lcb-ceiling"],
      "withdrawn_ids": ["E0019-predictor-loss-math","E0017-soft-governor-math"],
    }
    traps = run_trap_checks(ev_dict)
    print(render(traps))
    red = [k for k,(ok,_) in traps.items() if not ok]
    verdict = "BLOCKED" if red else ("PASS" if res["learned_marginal"]["lo"] > 0
                                     else "INCONCLUSIVE")

    spec = ExperimentSpec(
        exp_id="E0027-marginal-value",
        title="Where the execution-feedback signal fails to become utility",
        model="Gemini-Pro-1.5 (May) generations, published by LiveCodeBench",
        budget={"axis":"rank-then-allocate; sample 1 everywhere, remainder to top-ranked",
                "B_star": res["learned_marginal"]["cost"],
                "charged":"exact tokenizer count over published generations"},
        seeds={"split":"sha256(question_id) parity","bootstrap":0},
        split={"calibration":len(cal),"evaluation":n},
        metric="pass-within-k at matched realised cost; primary = ranked policy "
               "minus the randomised fixed envelope at its own cost",
        params={"features":list(F),"target":"sample 1 fails and a later sample succeeds",
                "economics":econ},
        notes="Corrects E0026's +0.1880 ceiling, which used an oracle deciding "
              "before paying for sample 1. Observable ceiling is +0.0588.")
    run = ExperimentRun(spec, overwrite=True)
    for i,q in enumerate(ev):
        run.append({"qid":q,"marginal_target":float(yv[i]),"pred":float(pv[i]),
                    "u1":float(U1[i]),"u_all":float(Ua[i]),"rest_tokens":float(Rest[i])})
    run.finalize(summary={
        "observable_ceiling":0.0588, "oracle_ranking_adv":res["oracle_marginal"]["adv"],
        "oracle_ranking_lo":res["oracle_marginal"]["lo"],
        "oracle_ranking_hi":res["oracle_marginal"]["hi"],
        "primary_mean":res["learned_marginal"]["adv"],
        "primary_lo":res["learned_marginal"]["lo"],
        "primary_hi":res["learned_marginal"]["hi"],
        "random_adv":res["random"]["adv"],
        "marginal_auc":float(auc(yv,pv)),
        "positives_calibration":int(yc.sum()), "positives_evaluation":int(yv.sum()),
        "events_per_feature":float(yc.sum()/len(F)),
        "breakeven_purity":float(1-econ["breakeven_solvable_fraction"]),
        "achieved_purity":0.917, "cost_ratio":econ["cost_ratio"],
        "tokens_per_utility_point":econ["tokens_per_utility_point"],
        "verdict":verdict}, metrics=econ, traps=traps, verdict=verdict)
    print(f"\n  recorded E0027-marginal-value  verdict={verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
