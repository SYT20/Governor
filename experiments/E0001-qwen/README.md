# E0001-qwen — Phase 4 reasoning curve and frozen engine/mode selection

**Verdict: PASS**

| field | value |
|---|---|
| model | `qwen/qwen3.6-27b` |
| commit | `8bbebc33b06509a34dacdb1ff76b3236ef38ffd0` |
| budget | `{"grid_max_tokens": [300, 700, 1400, 2800], "charged": "usage.total_tokens"}` |
| seeds | `{"calibration_pool": 1000}` |
| split | `{"calibration_items": 40, "test": "not touched"}` |
| raw rows | 160 (`raw.jsonl`, sha256 `dea3e3c1924faa56`) |
| wall | 1.3 s |

## Metric

mean exact-integer correctness over calibration items at each max_tokens budget; engine/mode chosen by the frozen rule in PREREGISTRATION-phase4-nemotron.md

## Summary

```json
{
  "chosen_engine": "qwen",
  "chosen_model": "qwen/qwen3.6-27b",
  "selection": {
    "high": 2800,
    "low": 700,
    "acc_high": 1.0,
    "acc_low": 0.525,
    "gap": 0.475,
    "qualifies": true,
    "episode_budget": 7512
  },
  "qualified": [
    "qwen"
  ]
}
```

## Trap checks

- GREEN `secret_scan` — files_with_credentials=[]

## Notes

Every (item, budget) pair is called once and cached; all later policies read the same responses (common random numbers).
