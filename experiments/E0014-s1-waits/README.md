# E0014-s1-waits — Fully-consumed reasoning units on external s1 data

**Verdict: UNIT-BINDS-NO-HEADROOM**

| field | value |
|---|---|
| model | `s1-32B via simplescaling/results (forcingignore{N}wait)` |
| commit | `d5e69bc198b3dc59910d2be0f3a9405e1a5a67d6` |
| budget | `{"unit": "forced Wait continuation", "levels": [1, 2, 3, 4, 5, 6, 7, 8], "charged": "exact tokens, simplescaling/s1-32B tokenizer"}` |
| seeds | `{"split": "n/a - screen only"}` |
| split | `{"items": 500, "benchmark": "MATH-500"}` |
| raw rows | 36 (`raw.jsonl`, sha256 `773a8f5de7e0cbfd`) |
| wall | 2.1 s |

## Metric

p(help) = fraction of items correct at high waits and wrong at low; ideal ceiling from the closed-form headroom law at n=12

## Summary

```json
{
  "verdict": "UNIT-BINDS-NO-HEADROOM",
  "n_pairs": 28,
  "n_pass_S1": 0,
  "best_ideal": 0.00863459402365589,
  "best_p_help": 0.01,
  "accuracy_at_1_wait": 0.928,
  "accuracy_at_8_waits": 0.918
}
```

## Trap checks

- GREEN `secret_scan` — files_with_credentials=[]

## Notes

Token counts are exact, not len/4. The E0013 approximation was off 25-35% and varied with N, so it is used for nothing here.
