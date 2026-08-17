# E0016-soft-budget-ceiling — Soft expected-budget ceiling on external s1 data

**Verdict: SOFT-BUDGET-HEADROOM**

| field | value |
|---|---|
| model | `s1-32B via simplescaling/results` |
| commit | `7c141e058f570232002bbdb79ce6d5db1b118955` |
| budget | `{"contract": "SOFT_EXPECTED_BUDGET", "definition": "E[sum actual generation tokens] <= B", "levels": [500, 1000, 2000, 4000, 8000, 16000, 32000], "charged": "simplescaling/s1-32B tokenizer (exact)"}` |
| seeds | `{"split": "n/a - ceiling screen over all items"}` |
| split | `{"math": 500, "gpqa": 198}` |
| raw rows | 50 (`raw.jsonl`, sha256 `fbe129dab4316618`) |
| wall | 1.2 s |

## Metric

U(multiple-choice-knapsack oracle) - U(best fixed policy, randomised between adjacent levels to match the expected budget exactly), both at identical E[tokens]

## Summary

```json
{
  "verdict": "SOFT-BUDGET-HEADROOM",
  "benchmarks_with_headroom": [
    "math",
    "gpqa"
  ],
  "max_ceiling_math": 0.1703452535335831,
  "max_ceiling_gpqa": 0.2680974525299935,
  "at_tokens_math": 813.0578333333333,
  "at_tokens_gpqa": 1535.8207070707072
}
```

## Trap checks

- GREEN `secret_scan` — files_with_credentials=[]
- GREEN `exact_token_counts` — token_cost_source='simplescaling/s1-32B tokenizer (exact)'

## Notes

No API calls. Runs BEFORE any controller, per the project rule that the ceiling is measured first.
