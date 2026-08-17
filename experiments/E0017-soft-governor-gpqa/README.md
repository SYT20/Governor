# E0017-soft-governor-gpqa — Soft-budget learned Governor on external gpqa

**Verdict: FAIL**

| field | value |
|---|---|
| model | `s1-32B via simplescaling/results` |
| commit | `10e7a4630752732ff953561706fdc769980795a5` |
| budget | `{"contract": "SOFT_EXPECTED_BUDGET", "grid": [1439.4689393939393, 1782.1843434343434, 2124.8997474747475, 2467.6151515151514, 2810.3305555555557, 3153.04595959596, 3495.761363636364], "reported_at": 1439.4689393939393, "charged": "simplescaling/s1-32B tokenizer (exact)"}` |
| seeds | `{"split": "doc_id parity: even=calibration, odd=evaluation (declared in E0013)", "bootstrap": 0}` |
| split | `{"calibration_items": 99, "evaluation_items": 99, "rule": "doc_id parity: even=calibration, odd=evaluation (declared in E0013)"}` |
| raw rows | 7 (`raw.jsonl`, sha256 `2cb6efd87f5a672d`) |
| wall | 0.1 s |

## Metric

mean correctness at matched expected tokens; primary = Governor minus the randomised-envelope fixed baseline; secondary = Governor minus myopic; 95% CI from a bootstrap over evaluation items with predictors and knobs frozen

## Summary

```json
{
  "benchmark": "gpqa",
  "reported_budget": 1439.4689393939393,
  "primary_mean": 0.01496166941012253,
  "primary_lo": -0.08147481073602297,
  "primary_hi": 0.09225951383392207,
  "primary_beats": false,
  "secondary_mean": 0.03878787878787879,
  "secondary_lo": -0.020454545454545458,
  "secondary_hi": 0.09090909090909094,
  "secondary_beats": false,
  "verdict": "FAIL",
  "architecture_justified": false
}
```

## Trap checks

- GREEN `oracle_leakage` — forbidden_features=[]
- GREEN `progress_as_cognition` — unjustified_progress_features=[]
- GREEN `split_leakage` — selection=99 evaluation=99 overlap=0
- GREEN `exact_token_counts` — token_cost_source='simplescaling/s1-32B tokenizer (exact)'
- GREEN `secret_scan` — files_with_credentials=[]

## Notes

Predictors and both knobs fitted/tuned on calibration only.
