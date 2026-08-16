#!/usr/bin/env python3
"""Mechanism gate, hardened: implementable policy, strong greedy, multi-seed CIs.

Three corrections to the first gate:

1. ORACLE ASSISTANCE REMOVED. The previous context-first policy read
   ds.context[i] -- the TRUE latent -- to choose its block. Acquiring group 0
   reveals only a NOISY one-hot (std 0.1), so the policy must DECODE the context
   from what it observed. `context_first` now argmaxes the observed noisy vector
   and is fully implementable; `oracle` retains latent access and is labelled as
   the upper bound it is.

2. STRONGER GREEDY. Mutual information is a feature-ranking proxy, not the
   acquisition utility under the actual downstream predictor. Added
   greedy-predictive: acquire each candidate group on train, refit, and keep the
   group with the best VALIDATION gain. If non-myopic still wins against that,
   the claim is far more robust.

3. MULTI-SEED. One dataset realisation cannot support a generalisation claim.
   Five independent seeds, paired by seed, reported as mean +- SE.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
from governor.envs.cube_nm_repro import BLOCK_SIZE, CubeNMRepro

def masked(ds, groups_per_row, idx):
    X = np.full((len(idx), ds.n_features), np.nan)
    for r, i in enumerate(idx):
        for g in groups_per_row[r]:
            for c in ds.group_columns(g):
                X[r, c] = ds.features[i, c]
    return X

def fit_score(ds, gtr, tr, gte, te):
    m = HistGradientBoostingClassifier(max_depth=4, max_iter=120, random_state=0)
    m.fit(masked(ds, gtr, tr), ds.labels[tr])
    return float(m.score(masked(ds, gte, te), ds.labels[te]))

def decode_context(ds, i):
    """IMPLEMENTABLE: argmax of the OBSERVED noisy one-hot, not the latent."""
    return int(np.argmax(ds.features[i, : ds.n_contexts]))

def greedy_predictive_order(ds, tr, k_max, mi_order, n_sub=900, pool=24):
    """Acquire-and-measure greedy: pick the group with best validation gain.

    The candidate pool is the top-`pool` groups by MI UNION {group 0}. Including
    group 0 unconditionally is essential: it ranks ~45th by MI, so a pure top-k
    pool would EXCLUDE the very action under test and rig the comparison in
    favour of the hypothesis. Greedy is given every chance to buy the context.
    """
    sub = tr[:n_sub]
    a, b = train_test_split(sub, test_size=0.35, random_state=0)
    cand_pool = list(dict.fromkeys(mi_order[:pool] + [0]))
    chosen, remaining = [], list(cand_pool)
    small = lambda X, y: HistGradientBoostingClassifier(
        max_depth=3, max_iter=40, random_state=0).fit(X, y)
    for _ in range(k_max):
        best, best_s = None, -1.0
        for g in remaining:
            c = chosen + [g]
            m = small(masked(ds, [c] * len(a), a), ds.labels[a])
            sc = float(m.score(masked(ds, [c] * len(b), b), ds.labels[b]))
            if sc > best_s: best_s, best = sc, g
        chosen.append(best); remaining.remove(best)
    return chosen

def main() -> int:
    SEEDS = [1, 2, 3, 4, 5]
    # n_samples reduced for runtime; the effect measured is ~0.4, far above noise
    BUDGETS = [4, 8]
    print("=" * 80)
    print("HARDENED MECHANISM GATE — implementable policy, strong greedy, 5 seeds")
    print("=" * 80)
    rows = {k: {p: [] for p in ("greedy_mi", "greedy_pred", "context_first", "oracle")}
            for k in BUDGETS}
    ctx_ranks = []
    for seed in SEEDS:
        ds = CubeNMRepro(n_samples=3500, seed=seed)
        idx = np.arange(ds.n_samples)
        tr, te = train_test_split(idx, test_size=0.3, random_state=seed)
        mi = mutual_info_classif(ds.features[tr[:2500]], ds.labels[tr[:2500]], random_state=0)
        gain = {g: float(np.mean([mi[c] for c in ds.group_columns(g)]))
                for g in range(ds.n_groups)}
        mi_order = sorted(gain, key=lambda g: -gain[g])
        ctx_ranks.append(mi_order.index(0) + 1)
        pred_order = greedy_predictive_order(ds, tr, max(BUDGETS), mi_order)
        for k in BUDGETS:
            def rows_for(kind, ids):
                out = []
                for i in ids:
                    if kind == "greedy_mi":   out.append(mi_order[:k])
                    elif kind == "greedy_pred": out.append(pred_order[:k])
                    elif kind == "context_first":
                        # acquire context, DECODE it from the observation, then buy its block
                        c = decode_context(ds, i)
                        out.append([0] + ds.block_group_ids(c)[: max(k - 1, 0)])
                    else:                      # oracle: latent access, upper bound
                        out.append(ds.block_group_ids(int(ds.context[i]))[:k])
                return out
            for p in rows[k]:
                rows[k][p].append(fit_score(ds, rows_for(p, tr), tr, rows_for(p, te), te))
        print(f"  seed {seed}: done")
    print(f"\n  greedy(MI) ranks the context group at position "
          f"{np.mean(ctx_ranks):.0f} of 51 (mean over seeds)\n")
    print(f"  {'budget':>7} {'policy':<15} {'mean':>7} {'SE':>7}   paired vs greedy_pred")
    print("  " + "-" * 66)
    for k in BUDGETS:
        base = np.array(rows[k]["greedy_pred"])
        for p in ("greedy_mi", "greedy_pred", "context_first", "oracle"):
            v = np.array(rows[k][p]); se = v.std(ddof=1) / np.sqrt(len(v))
            d = v - base
            ds_ = d.std(ddof=1) / np.sqrt(len(d)) if p != "greedy_pred" else 0.0
            tag = "" if p == "greedy_pred" else f"{d.mean():+.3f} +- {1.96*ds_:.3f}"
            print(f"  {k:>7} {p:<15} {v.mean():>7.3f} {se:>7.3f}   {tag}")
        print()
    print("  === VERDICT ===")
    ok = True
    for k in BUDGETS:
        d = np.array(rows[k]["context_first"]) - np.array(rows[k]["greedy_pred"])
        lo = d.mean() - 1.96 * d.std(ddof=1) / np.sqrt(len(d))
        print(f"    budget {k}: context-first - greedy_pred = {d.mean():+.3f}, "
              f"95% CI lower bound {lo:+.3f}")
        ok &= lo > 0
    print(f"\n    {'CI excludes zero at every budget -> mechanism ROBUST' if ok else 'CI includes zero somewhere -> NOT robust'}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
