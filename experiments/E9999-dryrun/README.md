# E9999-dryrun — script-path validation

**Verdict: DRYRUN**

| field | value |
|---|---|
| model | `qwen/qwen3.6-27b` |
| commit | `e40345b1cce28a0b0451f7effac146337eda56e5` |
| budget | `{"episode_total_tokens": 5312}` |
| seeds | `{"cal_pool": 1000}` |
| split | `{"note": "both splits = calibration pool; numbers are meaningless"}` |
| raw rows | 110 (`raw.jsonl`, sha256 `04a4aba9337e5878`) |
| wall | 0.1 s |

## Metric

none; plumbing check

## Summary

```json
{
  "note": "plumbing only",
  "mean": -0.011363636363636364,
  "lo": -0.05054868861011459,
  "hi": 0.02782141588284186,
  "beats": false,
  "loses": false
}
```

## Trap checks

- GREEN `greedy_collapse` — mean_delta=-0.011364 calls_identical=False
- GREEN `constant_schedule` — distinct_decision_patterns=5
- GREEN `oracle_leakage` — forbidden_features=[]
- GREEN `answered_vs_utility` — answered=1.0000 utility=0.6705 -> ok
- GREEN `token_accounting` — charged==used:True over_budget_calls=0
- GREEN `execution_vs_scoring` — scored_through_canonical_executor=True
- GREEN `progress_as_cognition` — unjustified_progress_features=[]
- GREEN `invariant_as_intelligence` — cells_with_constant_decision=5/7
- GREEN `frozen_before_heldout` — froze=cf4f6c4d heldout=xxxxxxxx
- GREEN `secret_scan` — files_with_credentials=[]

## Notes


