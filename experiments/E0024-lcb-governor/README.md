# E0024-lcb-governor — Governor on LiveCodeBench sample allocation (architecture reused)

**Verdict: BLOCKED**

| field | value |
|---|---|
| model | `Gemini-Pro-1.5 (May) generations, published by LiveCodeBench` |
| commit | `cca15fabe3ae2de4489b6e5d6e217e8d9f1e5391` |
| budget | `{"axis": "samples k in 1..10", "B_star": 153.48846666666665, "contract": "SOFT_EXPECTED_BUDGET + hard runtime cap", "charged": "exact tokenizer count over published LiveCodeBench generations"}` |
| seeds | `{"split": "sha256(question_id) parity", "bootstrap": 0}` |
| split | `{"calibration": 210, "evaluation": 190}` |
| raw rows | 190 (`raw.jsonl`, sha256 `6591a9274ba43e21`) |
| wall | 0.1 s |

## Metric

pass@k-style utility at matched realised cost; primary = Governor minus the randomised fixed envelope at the Governor's own cost; secondary = Governor minus myopic

## Summary

```json
{
  "governor_U": 0.3736842105263158,
  "myopic_U": 0.3736842105263158,
  "fixed_matched": 0.37373385687666427,
  "oracle": 0.37894736842105264,
  "governor_tokens": 128.8421052631579,
  "B_star": 153.48846666666665,
  "primary_mean": -4.973870632859506e-05,
  "primary_lo": -0.00024407986546674533,
  "primary_hi": 0.0,
  "secondary_mean": 0.0,
  "secondary_lo": 0.0,
  "secondary_hi": 0.0,
  "mcnemar_p": 1.0,
  "n_disagree": 1,
  "n_required_gov_vs_myopic": Infinity,
  "verdict": "INCONCLUSIVE"
}
```

## Trap checks

- **RED** `oracle_leakage` — forbidden_features=['difficulty_ord']
- GREEN `exact_token_counts` — token_cost_source='exact tokenizer count over published LiveCodeBench generations'
- GREEN `split_leakage` — selection=210 evaluation=190 overlap=0
- GREEN `budget_adherence` — realised=129 budget=153 over=-16.1% baseline=129 baseline_short_by=+0.0%
- GREEN `secret_scan` — files_with_credentials=[]

## Notes

softbudget.py reused unchanged; only the family adapter is new.
