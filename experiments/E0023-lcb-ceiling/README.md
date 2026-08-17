# E0023-lcb-ceiling — LiveCodeBench sample-allocation ceiling (no API calls)

**Verdict: CEILING-PASS**

| field | value |
|---|---|
| model | `Gemini-Pro-1.5 (May) generations, published by LiveCodeBench` |
| commit | `83548521500ea5e9e31fe9a50ae0bcda4cc4ce99` |
| budget | `{"axis": "number of samples k in 1..10", "contract": "SOFT_EXPECTED_BUDGET", "charged": "exact tokenizer count over the published generations"}` |
| seeds | `{"split": "not yet split - ceiling screen only"}` |
| split | `{"problems": 400, "samples_per_problem": 10}` |
| raw rows | 12 (`raw.jsonl`, sha256 `9730dd1d702fd029`) |
| wall | 1.4 s |

## Metric

U(multiple-choice-knapsack oracle) - U(best fixed k, randomised between adjacent k to match the expected budget), at identical expected tokens

## Summary

```json
{
  "verdict": "CEILING-PASS",
  "max_ceiling": 0.0572506425083526,
  "at_tokens": 219.49824999999998,
  "n_problems": 400,
  "never_pass": 0.545,
  "always_pass": 0.305,
  "mixed": 0.15
}
```

## Trap checks

- GREEN `exact_token_counts` — token_cost_source='exact tokenizer count over the published generations'
- GREEN `secret_scan` — files_with_credentials=[]

## Notes

Allocation axis changed, benchmark not merely enlarged. MATH-500 was closed by E0022 as statistically unsettleable.
