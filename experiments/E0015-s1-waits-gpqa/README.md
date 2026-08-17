# E0015-s1-waits-gpqa — Fully-consumed reasoning units on external GPQA-Diamond

**Verdict: UNIT-BINDS-NO-HEADROOM**

| field | value |
|---|---|
| model | `s1-32B via simplescaling/results (forcingignore{N}wait)` |
| commit | `4121a4b301cba1a8210de34a2931445c411468cd` |
| budget | `{"unit": "forced Wait continuation", "levels": [1, 2, 3, 4, 5, 6, 7, 8], "charged": "exact tokens, simplescaling/s1-32B tokenizer"}` |
| seeds | `{"split": "n/a - screen only"}` |
| split | `{"items": 198, "benchmark": "GPQA-Diamond"}` |
| raw rows | 36 (`raw.jsonl`, sha256 `1e44762e64a2720a`) |
| wall | 1.2 s |

## Metric

p(help) and the closed-form ideal ceiling at n=12

## Summary

```json
{
  "verdict": "UNIT-BINDS-NO-HEADROOM",
  "n_pass_S1": 0,
  "best_ideal": 0.016696956385891085,
  "best_p_help": 0.020202020202020204,
  "accuracy_1_wait": 0.5959595959595959,
  "accuracy_8_waits": 0.5909090909090909,
  "hurt_exceeds_help_pairs": 19,
  "n_pairs": 28
}
```

## Trap checks

- GREEN `secret_scan` — files_with_credentials=[]

## Notes

Run because MATH-500 might have been saturated. It was not the cause.
