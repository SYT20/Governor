#!/usr/bin/env python3
"""THE MECHANISM GATE: does myopic acquisition actually lose to non-myopic?

Everything in this project has been building toward one falsifiable question.
CUBE-NM is constructed so the answer should be yes: the context group carries zero
mutual information with the label, so a greedy acquirer will never buy it, yet it
is the only thing identifying which block holds the signal.

Policies, all under an identical hard budget of k acquisition groups:
  random        pick groups uniformly
  greedy        pick the group with the highest IMMEDIATE predictive gain
  context-first the non-myopic path: buy the context, then its block
  oracle        context known for free; buy only the informative block

If greedy >= context-first, the mechanism does not exist even in an environment
purpose-built to contain it, and this whole research direction is wrong.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from governor.envs.cube_nm_repro import BLOCK_SIZE, CubeNMRepro

def masked(ds, groups_per_row, Xidx):
    """Build the observed matrix: unacquired columns are NaN (missing)."""
    X = np.full((len(Xidx), ds.n_features), np.nan)
    for r, i in enumerate(Xidx):
        for g in groups_per_row[r]:
            for c in ds.group_columns(g):
                X[r, c] = ds.features[i, c]
    return X

def score(ds, groups_per_row_tr, tr, groups_per_row_te, te):
    Xtr, Xte = masked(ds, groups_per_row_tr, tr), masked(ds, groups_per_row_te, te)
    m = HistGradientBoostingClassifier(max_depth=4, max_iter=120, random_state=0)
    m.fit(Xtr, ds.labels[tr])
    return float(m.score(Xte, ds.labels[te]))

def main() -> int:
    ds = CubeNMRepro(n_samples=8000, seed=123)
    idx = np.arange(ds.n_samples)
    tr, te = train_test_split(idx, test_size=0.3, random_state=0)
    rng = np.random.default_rng(0)

    # greedy needs per-group immediate gain, estimated on TRAIN only
    from sklearn.feature_selection import mutual_info_classif
    sub = tr[:3000]
    mi = mutual_info_classif(ds.features[sub], ds.labels[sub], random_state=0)
    gain = {}
    for g in range(ds.n_groups):
        gain[g] = float(np.mean([mi[c] for c in ds.group_columns(g)]))
    greedy_order = sorted(gain, key=lambda g: -gain[g])
    ctx_rank = greedy_order.index(0)
    print("=" * 78)
    print("MECHANISM GATE — myopic vs non-myopic acquisition on cube_nm_repro")
    print("=" * 78)
    print(f"\n  immediate gain of the CONTEXT group : {gain[0]:.5f}")
    print(f"  mean immediate gain of block groups : "
          f"{np.mean([gain[g] for g in range(1, ds.n_groups)]):.5f}")
    print(f"  greedy would rank context at position {ctx_rank+1} of {ds.n_groups}")
    print(f"  -> a myopic acquirer {'IGNORES' if ctx_rank > 5 else 'buys'} the context\n")

    print(f"  {'budget k':>9} {'random':>9} {'greedy':>9} {'context-first':>14} {'oracle':>9}")
    print("  " + "-" * 56)
    rows = []
    for k in (2, 4, 6, 8, 11):
        def per_row(kind, ids):
            out = []
            for i in ids:
                if kind == "random":
                    out.append(list(rng.choice(ds.n_groups, size=min(k, ds.n_groups), replace=False)))
                elif kind == "greedy":
                    out.append(greedy_order[:k])
                elif kind == "context":      # 1 group for context, rest on its block
                    b = ds.block_group_ids(int(ds.context[i]))[: max(k - 1, 0)]
                    out.append([0] + b)
                else:                        # oracle: context free, all budget on the block
                    out.append(ds.block_group_ids(int(ds.context[i]))[:k])
            return out
        vals = {}
        for kind in ("random", "greedy", "context", "oracle"):
            vals[kind] = score(ds, per_row(kind, tr), tr, per_row(kind, te), te)
        rows.append((k, vals))
        print(f"  {k:>9} {vals['random']:>9.3f} {vals['greedy']:>9.3f} "
              f"{vals['context']:>14.3f} {vals['oracle']:>9.3f}")

    gaps = [v["context"] - v["greedy"] for _, v in rows]
    print(f"\n  === VERDICT ===")
    print(f"    mean (context-first - greedy) = {np.mean(gaps):+.3f}")
    print(f"    max  (context-first - greedy) = {np.max(gaps):+.3f}")
    if np.mean(gaps) > 0.05:
        print(f"    -> NON-MYOPIC ACQUISITION WINS. The mechanism exists and is")
        print(f"       measurable. This is the verified laboratory the project needed.")
    else:
        print(f"    -> greedy is not beaten even here. The research direction fails.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
