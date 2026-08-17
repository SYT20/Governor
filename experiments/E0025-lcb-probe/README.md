# E0025-lcb-probe — FINAL: early-generation signal for sample allocation

**Verdict: BLOCKED**

| field | value |
|---|---|
| model | `Gemini-Pro-1.5 (May) generations, published by LiveCodeBench` |
| commit | `19555afb8e6f0b8deb3fbcaf213051efb226147f` |
| budget | `{"axis": "samples k in 2..10, probe=2 always paid", "B_star": 279.16776666666664, "charged": "exact tokenizer count over published LiveCodeBench generations"}` |
| seeds | `{"split": "sha256(question_id) parity", "bootstrap": 0}` |
| split | `{"calibration": 210, "evaluation": 190}` |
| raw rows | 190 (`raw.jsonl`, sha256 `6cb30c53b8d183b5`) |
| wall | 0.1 s |

## Metric

pass-within-k at matched realised cost; primary = Governor minus the randomised fixed envelope at the Governor's own cost

## Summary

```json
{
  "governor_U": 0.4,
  "myopic_U": 0.3894736842105263,
  "fixed_matched": 0.3952423737682226,
  "oracle": 0.45263157894736844,
  "ceiling": 0.057389205179145864,
  "governor_tokens": 293.7368421052632,
  "B_star": 279.16776666666664,
  "primary_mean": 0.004624144655613056,
  "primary_lo": -0.004717944236112766,
  "primary_hi": 0.01711033066462389,
  "secondary_mean": 0.010363157894736841,
  "secondary_lo": 0.0,
  "secondary_hi": 0.026315789473684237,
  "n_disagree": 51,
  "verdict": "INCONCLUSIVE",
  "hard_stop": true
}
```

## Trap checks

- GREEN `oracle_leakage` — forbidden_features=[]
- GREEN `exact_token_counts` — token_cost_source='exact tokenizer count over published LiveCodeBench generations'
- GREEN `split_leakage` — selection=210 evaluation=190 overlap=0
- **RED** `budget_adherence` — realised=294 budget=279 over=+5.2% baseline=294 baseline_short_by=+0.0%
- GREEN `secret_scan` — files_with_credentials=[]

## Notes

Sample correctness never enters a feature; only generation text and cross-sample agreement. Hard stop on a non-PASS.
