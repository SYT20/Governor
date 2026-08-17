# E0012-governor-phase5 — Phase 5: Governor vs best fixed policy at equal tokens

**Verdict: PASS**

| field | value |
|---|---|
| model | `qwen/qwen3.6-27b` |
| commit | `cb6cd2691735b3942bb159e4ec161dc36e33ba95` |
| budget | `{"episode_total_tokens": 6486.0, "low": 300, "high": 700, "n_items": 12, "charged": "usage.total_tokens"}` |
| seeds | `{"pool": 1000, "cal_group": 11, "test_group": 22, "bootstrap": 0}` |
| split | `{"calibration_items": 40, "evaluation_items": 55, "frozen_by": "configs/phase4r_split.json", "independent_unit": "ITEM (4 episodes only; an episode-level CI would rest on four numbers)"}` |
| raw rows | 404 (`raw.jsonl`, sha256 `f3bfe3189b194ec3`) |
| wall | 0.2 s |

## Metric

mean fraction correct per episode; paired difference vs the calibration-frozen best fixed policy; 95% CI from a cluster bootstrap over items with the controller held fixed

## Summary

```json
{
  "primary_baseline": "fixed_best",
  "mean": 0.10807291666666669,
  "lo": 0.020833333333333315,
  "hi": 0.22916666666666666,
  "beats": true,
  "loses": false,
  "governor_U": 0.31385416666666666,
  "baseline_U": 0.20578125,
  "oracle_U": 0.3913541666666667,
  "headroom_captured": 0.5823744035924784,
  "ceiling_available": 0.18557291666666667
}
```

## Trap checks

- GREEN `greedy_collapse` — mean_delta=+0.187500 calls_identical=False
- GREEN `constant_schedule` — distinct_decision_patterns=4
- GREEN `oracle_leakage` — forbidden_features=[]
- GREEN `answered_vs_utility` — answered=1.0000 utility=0.3750 -> ok
- GREEN `token_accounting` — charged==used:True over_budget_calls=0
- GREEN `execution_vs_scoring` — scored_through_canonical_executor=True
- GREEN `progress_as_cognition` — unjustified_progress_features=[]
- GREEN `invariant_as_intelligence` — cells_with_constant_decision=6/12
- GREEN `frozen_before_heldout` — froze=aab0e2d3 heldout=run
- GREEN `split_leakage` — selection=40 evaluation=55 overlap=0
- GREEN `secret_scan` — files_with_credentials=[]

## Notes

Runs only because E0011 recorded CEILING-PASS on held-out items.
