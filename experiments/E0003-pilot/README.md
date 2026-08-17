# E0003-pilot — Phase 4 PILOT (underpowered): is there signal worth eight days of quota?

**Verdict: PILOT**

| field | value |
|---|---|
| model | `qwen/qwen3.6-27b` |
| commit | `a6c56c2f2998d02f0b545a02c933e54630d47f98` |
| budget | `{"episode_total_tokens": 5412.0, "low": 700, "high": 2800, "charged": "usage.total_tokens"}` |
| seeds | `{"pool": 1000, "split": "index parity", "bootstrap": 0}` |
| split | `{"cal_items": 44, "test_items": 44, "disjoint": true, "WARNING": "test items come from the CALIBRATION pool; this is NOT the preregistered held-out split"}` |
| raw rows | 311 (`raw.jsonl`, sha256 `fdd770c6124ce2cb`) |
| wall | 0.2 s |

## Metric

mean fraction correct per episode; paired difference vs the calibration-frozen best fixed policy, 95% CI from a cluster bootstrap over ITEMS (episodes are not independent here)

## Summary

```json
{
  "primary_baseline": "greedy",
  "mean": -0.06234848484848485,
  "lo": -0.13636363636363646,
  "hi": 0.0,
  "beats": false,
  "loses": false,
  "governor_U": 0.7272727272727273,
  "baseline_U": 0.7727272727272727,
  "oracle_U": 0.7954545454545454,
  "ceiling_available": 0.02356060606060606,
  "n_test_items": 44,
  "WARNING": "PILOT: not the preregistered test, not a Phase 4 result"
}
```

## Trap checks

- GREEN `greedy_collapse` — mean_delta=-0.045455 calls_identical=False
- GREEN `constant_schedule` — distinct_decision_patterns=7
- GREEN `oracle_leakage` — forbidden_features=[]
- GREEN `answered_vs_utility` — answered=1.0000 utility=0.7273 -> ok
- GREEN `token_accounting` — charged==used:True over_budget_calls=0
- GREEN `execution_vs_scoring` — scored_through_canonical_executor=True
- GREEN `progress_as_cognition` — unjustified_progress_features=[]
- GREEN `invariant_as_intelligence` — cells_with_constant_decision=4/7
- GREEN `frozen_before_heldout` — froze=407f8743 heldout=pilot
- GREEN `secret_scan` — files_with_credentials=[]

## Notes

PILOT. Underpowered by construction and recorded as such. The preregistered E0002 needs 420 items; Groq allows about 54/day.
