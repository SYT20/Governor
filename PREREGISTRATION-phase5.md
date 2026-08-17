# Preregistration — Phase 5: a design derived from the headroom law

**Written after the law was validated and before any Phase 5 number exists.**

Phase 4 and Phase 4R were both designed by intuition and rejected after their
ceilings were measured. `governor/phase4/headroom.py` derives the ceiling
instead, so a configuration can be screened before any quota is spent.

## The law

For binary per-item gains with `P(gain > 0) = p`, `n` items per episode and `k`
affordable upgrades:

```
ceiling(n, k, p) = ( E[min(k, X)] - k*p ) / n ,    X ~ Binomial(n, p)
                 ~ p(1-p) - 0.399*sqrt(p(1-p)/n)   at the optimal k = n*p
```

Closed form, validated against simulation to 4e-3. It says three things:

1. The ceiling peaks at `p = 0.5` and vanishes at both ends. A task where the
   deep budget almost always helps is as useless as one where it never does.
2. Episode length barely moves the ideal ceiling — 0.156 at n=4, 0.194 at n=12,
   asymptote 0.25.
3. **Episode length moves FRAGILITY, and that is what actually killed Phase 4R.**
   A one-unit drift in `k` costs 30% of the ceiling at n=4 and 5% at n=12.

## What the two failures were, quantitatively

| family | measured | ideal | realisation |
|---|---|---|---|
| Phase 4 | +0.0455 | +0.1152 | **39.5%** — environment destroyed it |
| Phase 4R, selection | +0.1389 | +0.1523 | **91.2%** — redesign worked |
| Phase 4R, held-out | +0.0417 | +0.0884 | 47.2% — `k` drifted 1.50 → 1.25 |

Phase 4R's structural fix succeeded. It failed the gate because its operating
point moved between splits, and at n=6 that halves the available ceiling.

## Frozen criteria

Three criteria, all checkable on cached data before spending anything.

**S1 — headroom.** `ceiling(n, k, p) >= 0.12`, computed by the law from the
measured `p`. This replaces the earlier `P(X>K) >= 0.60` heuristic, which was a
proxy for this quantity and admitted configurations with no headroom.

**S2 — decidability.** `mean(actual DEEP cost) / cap(DEEP) >= 0.70`. Unchanged;
it is what took realisation from 39.5% to 91.2%.

**S3 — stability (NEW, and the one Phase 4R lacked).** Both must hold:
- `ceiling(n, k-1, p) >= 0.80 * ceiling(n, k, p)` — a one-unit drift may not
  cost more than a fifth of the ceiling;
- `|k_selection - k_evaluation| <= 0.30` — the realised operating point must not
  move between splits.

## Sample size, from the measured bootstrap SD

The Phase 4R gate had SD 0.0424 at 24 items, and SD scales as `1/sqrt(N)`. To
clear a lower bound of 0.02:

| true ceiling | items needed |
|---|---|
| +0.152 (design realised at k) | 12 |
| +0.088 (k realised at k-1) | 36 |
| +0.042 (what drift produced) | 352 |

**The fix is the design, not the sample.** Buying 352 items costs two days of
quota; making `k` stable costs nothing. Phase 5 targets `N >= 48` evaluation
items, comfortably above the 36 needed even if `k` drops by one.

## Split

Unchanged and already frozen: `configs/phase4r_split.json`, by item id.
Selection = the first 40 items of pool seed 1000; evaluation = everything else.
`split_leakage` is a red trap. The gate reads evaluation items only.

## Gate

Unchanged: `U(clairvoyant) - U(budget-limited greedy) > 0.02` with a 95%
cluster-bootstrap lower bound above 0.02, resampling **items**, on evaluation
items only. `require_gate_passed()` blocks every downstream step until it
records `CEILING-PASS`.

## What would refute this

If a configuration satisfies S1, S2 and S3 and the held-out ceiling still fails,
then the law's assumption — that non-clairvoyant policies select items at random
with respect to gain — is wrong for this environment, and the right response is
to measure *that*, not to try a fourth family.
