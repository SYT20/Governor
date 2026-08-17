# E0021-enforced-gpqa — Budget-enforced Governor at matched realised cost (gpqa)

**Verdict: BLOCKED**

| field | value |
|---|---|
| model | `s1-32B via simplescaling/results` |
| commit | `f06b68ca9e53a2b95c9c8d55eb497cf6603c6058` |
| budget | `{"contract": "SOFT_EXPECTED_BUDGET + hard runtime cap", "B_star": 1276.7463636363636, "charged": "simplescaling/s1-32B tokenizer (exact)"}` |
| seeds | `{"split": "doc_id parity", "bootstrap": 0}` |
| split | `{"calibration_items": 99, "evaluation_items": 99}` |
| raw rows | 99 (`raw.jsonl`, sha256 `42afc08ca62df975`) |
| wall | 0.1 s |

## Metric

primary = Governor minus the fixed envelope AT THE GOVERNOR'S OWN REALISED COST; secondary = Governor minus myopic; 95% CI from a bootstrap over evaluation items with the controller frozen

## Summary

```json
{
  "benchmark": "gpqa",
  "B_star": 1276.7463636363636,
  "governor_U": 0.36363636363636365,
  "governor_tokens": 1325.8181818181818,
  "unenforced_tokens": 1325.8181818181818,
  "fixed_at_matched_cost": 0.35251751415737886,
  "primary_mean": 0.008612375373188368,
  "primary_lo": -0.027712073642014282,
  "primary_hi": 0.04093775762888517,
  "secondary_mean": 0.020247474747474747,
  "secondary_lo": 0.0,
  "secondary_hi": 0.0505050505050505,
  "mcnemar_p": 0.5,
  "n_disagree": 17,
  "verdict": "INCONCLUSIVE"
}
```

## Trap checks

- GREEN `oracle_leakage` — forbidden_features=[]
- GREEN `exact_token_counts` — token_cost_source='simplescaling/s1-32B tokenizer (exact)'
- GREEN `split_leakage` — selection=99 evaluation=99 overlap=0
- **RED** `budget_adherence` — realised=1326 budget=1277 over=+3.8% baseline=1326 baseline_short_by=+0.0%
- GREEN `secret_scan` — files_with_credentials=[]

## Notes

E0019 is withdrawn: it spent 15% over budget and was scored against a baseline at the nominal budget.
