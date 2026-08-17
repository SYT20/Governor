# E0016-soft-budget-ceiling — Soft expected-budget ceiling on external s1 data

**Verdict: SOFT-BUDGET-NO-HEADROOM**

| field | value |
|---|---|
| model | `s1-32B via simplescaling/results` |
| commit | `d5bf2c4013c8509f66ecd1de99ba17f44292ca1d` |
| budget | `{"contract": "SOFT_EXPECTED_BUDGET", "definition": "E[sum actual generation tokens] <= B", "levels": [500, 1000, 2000, 4000, 8000, 16000, 32000], "charged": "simplescaling/s1-32B tokenizer (exact)"}` |
| seeds | `{"split": "n/a - ceiling screen over all items"}` |
| split | `{"math": 500, "gpqa": 198}` |
| raw rows | 50 (`raw.jsonl`, sha256 `9999186750798de0`) |
| wall | 1.7 s |

## Metric

U(multiple-choice-knapsack oracle) - U(best fixed policy, randomised between adjacent levels to match the expected budget exactly), both at identical E[tokens]

## Summary

```json
{
  "verdict": "SOFT-BUDGET-NO-HEADROOM",
  "benchmarks_with_headroom": [],
  "max_ceiling_math": NaN,
  "max_ceiling_gpqa": NaN,
  "at_tokens_math": 521.918,
  "at_tokens_gpqa": 1251.7121212121212
}
```

## Trap checks

- GREEN `secret_scan` — files_with_credentials=[]
- GREEN `exact_token_counts` — token_cost_source='simplescaling/s1-32B tokenizer (exact)'

## Notes

No API calls. Runs BEFORE any controller, per the project rule that the ceiling is measured first.
