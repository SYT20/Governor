#!/usr/bin/env python3
"""INVALID AS WRITTEN -- DO NOT RUN. Kept as a specification of the right
experiment and a record of how it was wrong.

THE FLAW, found in review before any number was produced. The script advances
the episode trajectory using H at EVERY decision:

    g = ah                       # advance under H by default
    logL = logL + ib.b.loglik_cols(x, ib.group_cols[g])

then scores each policy by SUMMING precomputed local dstar values from that
single H trajectory. So no policy is ever executed. Three consequences:

  * the "oracle" picks the two largest local dstar on an H trajectory, which is
    an upper bound on a FIXED trajectory, not the optimum of the sequential
    problem -- spending M2 at t=1 changes s_2, and this assumes s_2^M2 = s_2^H
  * "GOVERNOR" predicts an allocation and is then scored by summing the same
    H-trajectory numbers, never running its own decisions
  * "greedy_M2" has the identical defect

And sum_t dstar_t is not the episode-level improvement unless the effects are
additive AND transitions are action-independent, neither of which holds here.
This is the same class of error as treating a convenient local quantity as the
system objective, which produced several earlier false positives in this
project.

WHAT THE CORRECT VERSION MUST DO. For each policy, reset the identical episode
and EXECUTE it: observe state, choose H or M2, run that mode, update the state
from the real outcome, decrement compute on M2, repeat for 4 decisions; then
record final task utility. With 4 decisions and 2 M2 calls the true oracle is an
exhaustive enumeration of the 11 allocation patterns, each actually executed --
no DP framework needed. All policies must share matched random draws per
episode, as in the CUBE-NM work.

The headline comparison is U_Governor - U_greedy_M2 (budget-limited
always-deliberate), NOT U_Governor - U_H.

Original docstring follows.

The experiment that tests the actual thesis: a BINDING compute budget.

Every prior result compared H against M2 at a single decision, where M2 was
always affordable. With no opportunity cost that is a selector, not a Governor,
and always-M2 was free to win -- which is why the previous test beat always-H by
+0.0347 but always-M2 by only +0.0035.

Here one episode has FOUR decision points sharing ONE compute envelope that
affords only TWO M2 invocations. Now spending M2 early forecloses it later and
the controller must answer "which decisions deserve the scarce reasoning?"

BASELINES, corrected under review. "Always-M2" is infeasible and must not be the
comparator:
  H          never invoke M2
  greedy_M2  invoke M2 at the first affordable decisions, then fall back to H
             -- the honest budget-limited version of always-deliberate
  regime_only allocate using ONLY the regime posterior. Separates state-level
             cognition from regime identification: if Governor ~ regime_only,
             the advantage is inferring which regime it is in, not reasoning
             about this state.
  oracle     allocate the two M2 calls to the states with the largest true
             Delta* -- the achievable ceiling
  GOVERNOR   allocate using observable state only

PREREGISTERED: train seed 7, test seed 31337 (fresh), sigma in {0.35,0.60,1.50}
at B_tool=6 with 4 decision points, B_compute = 2 M2 invocations. Features are
posterior-derived; no sigma, no configuration id.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sklearn.ensemble import HistGradientBoostingRegressor
from governor.envs.env5_modes import InstrumentedBayes, h_gate_first, m2_plan
from governor.envs.gated_family import N_LABELS
from governor.envs.probe_family import ObservableProbeBayes, ProbeTask, make_config

CELLS = ((0.35, 6.0), (0.60, 6.0), (1.50, 6.0))
N_EP, N_DEC, N_M2, K = 30, 4, 2, 96
FEAT = 8


def feats(ib, logL):
    p = ib.b._norm(logL); M = p.reshape(ib.b.R, ib.b.K, N_LABELS)
    py, pc, pr = M.sum(axis=(0,1)), M.sum(axis=(0,2)), M.sum(axis=(1,2))
    sy, sc, sr = np.sort(py)[::-1], np.sort(pc)[::-1], np.sort(pr)[::-1]
    e = lambda q: -float((q*np.log(np.maximum(q,1e-300))).sum())
    return [e(py), float(sy[0]), float(sy[0]-sy[1]), e(pc), float(sc[0]),
            e(pr), float(sr[0]), float(sr[0]-sr[1])]


def episode(ib, x, y, B):
    """Walk the episode, recording at each decision the H action, the M2 action,
    the observable features, and the true Delta* for spending M2 here."""
    logL, av, spent, seen = ib.prior_logL(), list(range(ib.n_groups)), 0.0, []
    recs = []
    for _ in range(N_DEC):
        rem = B - spent
        ah = h_gate_first(ib, logL, av, rem, acquired=bool(seen))
        am = m2_plan(ib, logL, av, rem)
        if ah is None or am is None:
            break
        p = ib.b._norm(logL); rng = np.random.default_rng(len(recs)*13 + 5)
        d = 0.0
        for _ in range(K):
            h = int(rng.choice(len(p), p=p))
            xs = ib.b.MU[h] + ib.b.SD[h]*rng.standard_normal(ib.b.nf)
            if seen: xs[seen] = x[seen]
            yy = h % N_LABELS
            d += int(_roll(ib, xs, logL, av, rem, am) == yy) \
                - int(_roll(ib, xs, logL, av, rem, ah) == yy)
        recs.append({"f": feats(ib, logL), "dstar": d/K, "ah": ah, "am": am,
                     "logL": logL.copy(), "av": list(av), "rem": rem})
        g = ah                                   # advance under H by default
        av.remove(g); spent += float(ib.cost[g]); seen += ib.group_cols[g]
        logL = logL + ib.b.loglik_cols(x, ib.group_cols[g])
    return recs


def _roll(ib, x, logL, av, B, first):
    logL, av, sp = logL.copy(), list(av), 0.0
    if first is not None:
        av.remove(first); sp += float(ib.cost[first])
        logL = logL + ib.b.loglik_cols(x, ib.group_cols[first])
    while True:
        g = ib.b.myopic_step(logL, av, B - sp)
        if g is None: break
        av.remove(g); sp += float(ib.cost[g])
        logL = logL + ib.b.loglik_cols(x, ib.group_cols[g])
    return int(np.argmax(ib.b.label_posterior(logL)))


def value(recs, which):
    """Total Delta* gained by spending the N_M2 budget on the chosen decisions."""
    return float(sum(recs[i]["dstar"] for i in which))


def collect(seed, tag):
    eps = []
    for so, B in CELLS:
        t = ProbeTask(cfg=make_config(so,1.0,0.05), n_samples=N_EP, seed=seed)
        ib = InstrumentedBayes(ObservableProbeBayes(t))
        for i in range(N_EP):
            r = episode(ib, t.features[i], int(t.labels[i]), B)
            if len(r) >= N_DEC: eps.append(r)
        print(f"  {tag}: sigma={so} done ({len(eps)} eps)", flush=True)
    return eps


def main() -> int:
    print("BINDING-BUDGET GOVERNOR TEST", flush=True)
    print(f"  {N_DEC} decisions/episode, compute affords {N_M2} M2 calls", flush=True)
    tr, te = collect(7, "train"), collect(31337, "test")
    Xtr = np.array([d["f"] for e in tr for d in e])
    ytr = np.array([d["dstar"] for e in tr for d in e])
    m_all = HistGradientBoostingRegressor(max_depth=3, max_iter=200, random_state=0).fit(Xtr, ytr)
    m_reg = HistGradientBoostingRegressor(max_depth=3, max_iter=200, random_state=0).fit(Xtr[:, 5:8], ytr)

    res = {k: [] for k in ("H","greedy_M2","regime_only","oracle","GOVERNOR")}
    for e in te:
        ds = np.array([d["dstar"] for d in e])
        F = np.array([d["f"] for d in e])
        res["H"].append(0.0)
        res["greedy_M2"].append(value(e, range(N_M2)))
        res["oracle"].append(value(e, np.argsort(-ds)[:N_M2]))
        res["GOVERNOR"].append(value(e, np.argsort(-m_all.predict(F))[:N_M2]))
        res["regime_only"].append(value(e, np.argsort(-m_reg.predict(F[:, 5:8]))[:N_M2]))
    print(f"\n  held-out episodes: {len(te)}", flush=True)
    for k in ("H","greedy_M2","regime_only","GOVERNOR","oracle"):
        v = np.array(res[k]); se = v.std(ddof=1)/np.sqrt(len(v))
        print(f"    {k:<12} {v.mean():+.4f} +- {1.96*se:.4f}", flush=True)
    g = np.array(res["GOVERNOR"])
    for base in ("greedy_M2","regime_only"):
        d = g - np.array(res[base]); se = d.std(ddof=1)/np.sqrt(len(d))
        ok = d.mean()-1.96*se > 0
        print(f"\n    GOVERNOR - {base}: {d.mean():+.4f} "
              f"[{d.mean()-1.96*se:+.4f}, {d.mean()+1.96*se:+.4f}]  "
              f"{'BEATS IT' if ok else 'not separable'}", flush=True)
    o = np.array(res["oracle"]); gm = np.array(res["greedy_M2"])
    print(f"\n    headroom over greedy_M2 captured: "
          f"{(g.mean()-gm.mean())/max(o.mean()-gm.mean(),1e-9):.0%}", flush=True)
    Path("results/env5_binding_budget.json").write_text(json.dumps(
        {k: [float(x) for x in v] for k, v in res.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
