# E0004-ceiling — Phase 4 observable ceiling: what could ANY allocator gain?

**Verdict: PREMISE-FAILS**

| field | value |
|---|---|
| model | `qwen/qwen3.6-27b` |
| commit | `b6df47d91949129938b47f680e75e56ef61d11e5` |
| budget | `{"swept": [3312, 3712, 4112, 4512, 4912, 5312, 5712, 6112, 6512, 6912, 7312, 7712, 8112, 8512, 8912, 9312, 9712, 10112, 10512, 10912, 11312, 11712], "low": 700, "high": 2800}` |
| seeds | `{"pool": 1000, "grouping": 7}` |
| split | `{"items": 88, "episodes": 22, "note": "calibration pool only"}` |
| raw rows | 22 (`raw.jsonl`, sha256 `88b40741153348e8`) |
| wall | 1.2 s |

## Metric

U(clairvoyant optimum) - U(budget-limited greedy), per episode budget, both executed through the canonical executor

## Summary

```json
{
  "max_ceiling": 0.045454545454545525,
  "at_budget": 5712,
  "verdict": "PREMISE-FAILS",
  "frac_benefiting": 0.5568181818181818,
  "cap_to_actual_high": 0.33486633755588674
}
```

## Trap checks

- GREEN `secret_scan` — files_with_credentials=[]

## Notes

The check that should have preceded building the controller.
