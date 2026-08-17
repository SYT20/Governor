# E0021-enforced-math — Budget-enforced Governor at matched realised cost (math)

**Verdict: INCONCLUSIVE**

| field | value |
|---|---|
| model | `s1-32B via simplescaling/results` |
| commit | `fa710d7d102d734787f8104866134a58c2b5254c` |
| budget | `{"contract": "SOFT_EXPECTED_BUDGET + hard runtime cap", "B_star": 845.7475505084745, "charged": "simplescaling/s1-32B tokenizer (exact)"}` |
| seeds | `{"split": "doc_id parity", "bootstrap": 0}` |
| split | `{"calibration_items": 250, "evaluation_items": 250}` |
| raw rows | 250 (`raw.jsonl`, sha256 `d64efaad5e4a0b23`) |
| wall | 0.3 s |

## Metric

primary = Governor minus the fixed envelope AT THE GOVERNOR'S OWN REALISED COST; secondary = Governor minus myopic; 95% CI from a bootstrap over evaluation items with the controller frozen

## Summary

```json
{
  "benchmark": "math",
  "B_star": 845.7475505084745,
  "governor_U": 0.796,
  "governor_tokens": 845.208,
  "unenforced_tokens": 973.484,
  "fixed_at_matched_cost": 0.7840296773469861,
  "primary_mean": 0.012147261036831973,
  "primary_lo": -0.039637056181682354,
  "primary_hi": 0.050996757923367525,
  "secondary_mean": 0.020217999999999996,
  "secondary_lo": -0.020000000000000018,
  "secondary_hi": 0.06000000000000005,
  "mcnemar_p": 0.42435622215270996,
  "n_disagree": 101,
  "verdict": "INCONCLUSIVE"
}
```

## Trap checks

- GREEN `oracle_leakage` — forbidden_features=[]
- GREEN `exact_token_counts` — token_cost_source='simplescaling/s1-32B tokenizer (exact)'
- GREEN `split_leakage` — selection=250 evaluation=250 overlap=0
- GREEN `budget_adherence` — realised=845 budget=846 over=-0.1% baseline=845 baseline_short_by=+0.0%
- GREEN `secret_scan` — files_with_credentials=[]

## Notes

E0019 is withdrawn: it spent 15% over budget and was scored against a baseline at the nominal budget.
