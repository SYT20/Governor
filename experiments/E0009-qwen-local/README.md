# E0009-qwen-local — Local Qwen backend reasoning curve (MLX)

**Verdict: CURVE-VALID**

| field | value |
|---|---|
| model | `mlx-community/Qwen3-1.7B-4bit` |
| commit | `8754ea59b001f46d51afb244a444b201d5c37241` |
| budget | `{"grid_max_tokens": [300, 700, 1400, 2800], "charged": "completion tokens"}` |
| seeds | `{"pool": 1000}` |
| split | `{"selection_items": 6}` |
| raw rows | 24 (`raw.jsonl`, sha256 `69fd258d2f908139`) |
| wall | 663.9 s |

## Metric

mean exact-integer correctness at each max_tokens budget, on the same calibration items as E0001

## Summary

```json
{
  "selection": {
    "high": 2800,
    "low": 1400,
    "acc_high": 0.6666666666666666,
    "acc_low": 0.5,
    "gap": 0.16666666666666663,
    "qualifies": true,
    "episode_budget": 8912
  },
  "verdict": "CURVE-VALID",
  "max_acc": 0.6666666666666666
}
```

## Trap checks

- GREEN `secret_scan` — files_with_credentials=[]

## Notes

Backend experiment. The Governor is unchanged.
