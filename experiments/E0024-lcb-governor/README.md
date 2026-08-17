# E0024-lcb-governor — Governor on LiveCodeBench sample allocation (architecture reused)

**Verdict: INCONCLUSIVE**

| field | value |
|---|---|
| model | `Gemini-Pro-1.5 (May) generations, published by LiveCodeBench` |
| commit | `aac6c87b8d7b903ba5cd5c7552c1d5c8ebad839d` |
| budget | `{"axis": "samples k in 1..10", "B_star": 153.48846666666665, "contract": "SOFT_EXPECTED_BUDGET + hard runtime cap", "charged": "exact tokenizer count over published LiveCodeBench generations"}` |
| seeds | `{"split": "sha256(question_id) parity", "bootstrap": 0}` |
| split | `{"calibration": 210, "evaluation": 190}` |
| raw rows | 190 (`raw.jsonl`, sha256 `7cf6e316726c03d6`) |
| wall | 0.1 s |

## Metric

pass@k-style utility at matched realised cost; primary = Governor minus the randomised fixed envelope at the Governor's own cost; secondary = Governor minus myopic

## Summary

```json
{
  "governor_U": 0.3736842105263158,
  "myopic_U": 0.3736842105263158,
  "fixed_matched": 0.37647407751278233,
  "oracle": 0.43157894736842106,
  "governor_tokens": 151.21052631578948,
  "B_star": 153.48846666666665,
  "primary_mean": -0.0027819797365463057,
  "primary_lo": -0.006584622967170927,
  "primary_hi": 0.0,
  "secondary_mean": 0.0,
  "secondary_lo": 0.0,
  "secondary_hi": 0.0,
  "mcnemar_p": 1.0,
  "n_disagree": 45,
  "n_required_gov_vs_myopic": Infinity,
  "verdict": "INCONCLUSIVE"
}
```

## Trap checks

- GREEN `oracle_leakage` — forbidden_features=[]
- GREEN `exact_token_counts` — token_cost_source='exact tokenizer count over published LiveCodeBench generations'
- GREEN `split_leakage` — selection=210 evaluation=190 overlap=0
- GREEN `budget_adherence` — realised=151 budget=153 over=-1.5% baseline=151 baseline_short_by=+0.0%
- GREEN `secret_scan` — files_with_credentials=[]

## Notes

softbudget.py reused unchanged; only the family adapter is new.
