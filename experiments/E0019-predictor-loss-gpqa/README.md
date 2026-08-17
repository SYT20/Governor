# E0019-predictor-loss-gpqa — Predictor loss and calibration, no probe (gpqa)

**Verdict: FAIL**

| field | value |
|---|---|
| model | `s1-32B via simplescaling/results` |
| commit | `e4ff975852b23ae32b04917e41f4df53a8752f07` |
| budget | `{"contract": "SOFT_EXPECTED_BUDGET", "reported_at": 1276.7463636363636, "charged": "simplescaling/s1-32B tokenizer (exact)"}` |
| seeds | `{"split": "doc_id parity", "bootstrap": 0}` |
| split | `{"calibration_items": 99, "evaluation_items": 99}` |
| raw rows | 28 (`raw.jsonl`, sha256 `3a229b540a753ba9`) |
| wall | 0.1 s |

## Metric

mean correctness at matched expected tokens; variants differ ONLY in the correctness predictor's loss and calibration; the variant is chosen by calibration-side Brier, never by outcome

## Summary

```json
{
  "benchmark": "gpqa",
  "selected_variant": "logistic+isotonic",
  "brier_by_variant": {
    "ridge": 0.20859498569480875,
    "logistic": 0.2050840306249239,
    "logistic+sigmoid": 0.22729267282112747,
    "logistic+isotonic": 0.1996193341301144
  },
  "ece_by_variant": {
    "ridge": 0.0854817004498095,
    "logistic": 0.06951069738124102,
    "logistic+sigmoid": 0.09361913583823135,
    "logistic+isotonic": 0.06808336416632758
  },
  "primary_mean": 0.00970797932391412,
  "primary_lo": -0.052510264788629434,
  "primary_hi": 0.0505050505050505,
  "secondary_mean": 0.020606060606060607,
  "secondary_lo": 0.0,
  "secondary_hi": 0.0505050505050505,
  "ceiling": 0.23232323232323238,
  "verdict": "FAIL"
}
```

## Trap checks

- GREEN `oracle_leakage` — forbidden_features=[]
- GREEN `exact_token_counts` — token_cost_source='simplescaling/s1-32B tokenizer (exact)'
- GREEN `split_leakage` — selection=99 evaluation=99 overlap=0
- GREEN `secret_scan` — files_with_credentials=[]

## Notes

No probe. The probe must earn its own token cost and that is a separate question.
