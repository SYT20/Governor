# E0021-enforced-gpqa — Budget-enforced Governor at matched realised cost (gpqa)

**Verdict: INCONCLUSIVE**

| field | value |
|---|---|
| model | `s1-32B via simplescaling/results` |
| commit | `4b9707e6c8072a20c07a5e533afc8de10b3437bf` |
| budget | `{"contract": "SOFT_EXPECTED_BUDGET + hard runtime cap", "B_star": 1276.7463636363636, "charged": "simplescaling/s1-32B tokenizer (exact)"}` |
| seeds | `{"split": "doc_id parity", "bootstrap": 0}` |
| split | `{"calibration_items": 99, "evaluation_items": 99}` |
| raw rows | 99 (`raw.jsonl`, sha256 `afa19f64e0785e66`) |
| wall | 0.1 s |

## Metric

primary = Governor minus the fixed envelope AT THE GOVERNOR'S OWN REALISED COST; secondary = Governor minus myopic; 95% CI from a bootstrap over evaluation items with the controller frozen

## Summary

```json
{
  "benchmark": "gpqa",
  "B_star": 1276.7463636363636,
  "governor_U": 0.3434343434343434,
  "governor_tokens": 1281.949494949495,
  "unenforced_tokens": 1325.8181818181818,
  "fixed_at_matched_cost": 0.3434343434343434,
  "primary_mean": 0.0,
  "primary_lo": 0.0,
  "primary_hi": 0.0,
  "secondary_mean": 0.0,
  "secondary_lo": 0.0,
  "secondary_hi": 0.0,
  "mcnemar_p": 1.0,
  "n_disagree": 0,
  "verdict": "INCONCLUSIVE"
}
```

## Trap checks

- GREEN `oracle_leakage` — forbidden_features=[]
- GREEN `exact_token_counts` — token_cost_source='simplescaling/s1-32B tokenizer (exact)'
- GREEN `split_leakage` — selection=99 evaluation=99 overlap=0
- GREEN `budget_adherence` — realised=1282 budget=1277 over=+0.4% baseline=1282 baseline_short_by=+0.0%
- GREEN `secret_scan` — files_with_credentials=[]

## Notes

E0019 is withdrawn: it spent 15% over budget and was scored against a baseline at the nominal budget.
