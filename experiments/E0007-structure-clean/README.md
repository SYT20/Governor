# E0007-structure-clean — Phase 4R structural search: which configuration can pose an allocation problem at all?

**Verdict: CONFIG-FOUND**

| field | value |
|---|---|
| model | `qwen/qwen3.6-27b` |
| commit | `44c1a46d45a95108e6c4ea91468da6d20d513c88` |
| budget | `{"swept": "per configuration", "charged": "usage.total_tokens"}` |
| seeds | `{"pool": 1000, "grouping": 7}` |
| split | `{"note": "calibration items only; the gate re-tests on held-out episodes"}` |
| raw rows | 57 (`raw.jsonl`, sha256 `cd498a317d5b326d`) |
| wall | 13.7 s |

## Metric

S1 = P(X>K) and E[X]/K where X is useful opportunities and K affordable upgrades; S2 = mean actual DEEP cost / cap(DEEP); ceiling = U(oracle) - U(greedy) through the canonical executor

## Summary

```json
{
  "verdict": "CONFIG-FOUND",
  "n_configs": 57,
  "n_satisfying_both": 1,
  "chosen": {
    "n_items": 6,
    "low": 300,
    "high": 700,
    "budget": 2868,
    "n_pool": 40,
    "n_episodes": 6,
    "p_useful_item": 0.475,
    "P_X_gt_K": 0.6666666666666666,
    "EX_over_K": 1.888888888888889,
    "act_over_cap": 0.8030797101449276,
    "mean_X": 2.8333333333333335,
    "mean_K": 1.5,
    "cheap": 0.05555555555555555,
    "greedy": 0.13888888888888887,
    "oracle": 0.2777777777777778,
    "ceiling": 0.13888888888888892,
    "S1": true,
    "S2": true
  }
}
```

## Trap checks

- GREEN `secret_scan` — files_with_credentials=[]
- GREEN `split_leakage` — selection=40 evaluation=360 overlap=0

## Notes

Configuration selected by S1 and S2, both frozen before this ran. The ceiling is reported for every cell, never used to select.
