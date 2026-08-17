# E0017-soft-governor-math — Soft-budget learned Governor on external math

**Verdict: FAIL**

| field | value |
|---|---|
| model | `s1-32B via simplescaling/results` |
| commit | `40fd685ef65daa3791de03da43e4ce0e5b61a29d` |
| budget | `{"contract": "SOFT_EXPECTED_BUDGET", "grid": [600.2057, 1002.1209166666667, 1404.0361333333335, 1805.95135, 2207.8665666666666, 2609.781783333334, 3011.697], "reported_at": 600.2057, "charged": "simplescaling/s1-32B tokenizer (exact)"}` |
| seeds | `{"split": "doc_id parity: even=calibration, odd=evaluation (declared in E0013)", "bootstrap": 0}` |
| split | `{"calibration_items": 250, "evaluation_items": 250, "rule": "doc_id parity: even=calibration, odd=evaluation (declared in E0013)"}` |
| raw rows | 7 (`raw.jsonl`, sha256 `6b15e1b228c5a4bd`) |
| wall | 0.1 s |

## Metric

mean correctness at matched expected tokens; primary = Governor minus the randomised-envelope fixed baseline; secondary = Governor minus myopic; 95% CI from a bootstrap over evaluation items with predictors and knobs frozen

## Summary

```json
{
  "benchmark": "math",
  "reported_budget": 600.2057,
  "primary_mean": 0.006920104286908016,
  "primary_lo": -0.014546622619192127,
  "primary_hi": 0.030171381489139573,
  "primary_beats": false,
  "secondary_mean": 0.023913333333333325,
  "secondary_lo": -0.0040000000000000036,
  "secondary_hi": 0.051999999999999935,
  "secondary_beats": false,
  "verdict": "FAIL",
  "architecture_justified": false
}
```

## Trap checks

- GREEN `oracle_leakage` — forbidden_features=[]
- GREEN `progress_as_cognition` — unjustified_progress_features=[]
- GREEN `split_leakage` — selection=250 evaluation=250 overlap=0
- GREEN `exact_token_counts` — token_cost_source='simplescaling/s1-32B tokenizer (exact)'
- GREEN `secret_scan` — files_with_credentials=[]

## Notes

Predictors and both knobs fitted/tuned on calibration only.
