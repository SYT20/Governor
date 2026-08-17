# E0019-predictor-loss-math — Predictor loss and calibration, no probe (math)

**Verdict: FAIL**

| field | value |
|---|---|
| model | `s1-32B via simplescaling/results` |
| commit | `a99dff4afabaa84f2633ea1cdf266032fdd7f965` |
| budget | `{"contract": "SOFT_EXPECTED_BUDGET", "reported_at": 600.2057, "charged": "simplescaling/s1-32B tokenizer (exact)"}` |
| seeds | `{"split": "doc_id parity", "bootstrap": 0}` |
| split | `{"calibration_items": 250, "evaluation_items": 250}` |
| raw rows | 28 (`raw.jsonl`, sha256 `5a09bc5ee2b1b194`) |
| wall | 0.1 s |

## Metric

mean correctness at matched expected tokens; variants differ ONLY in the correctness predictor's loss and calibration; the variant is chosen by calibration-side Brier, never by outcome

## Summary

```json
{
  "benchmark": "math",
  "selected_variant": "logistic+isotonic",
  "brier_by_variant": {
    "ridge": 0.1031069443542856,
    "logistic": 0.1025614654843342,
    "logistic+sigmoid": 0.10645121299835673,
    "logistic+isotonic": 0.09754714197497064
  },
  "ece_by_variant": {
    "ridge": 0.038783382515682883,
    "logistic": 0.04414660273451976,
    "logistic+sigmoid": 0.06937631194715257,
    "logistic+isotonic": 0.02957908757236396
  },
  "primary_mean": 0.01852908576231339,
  "primary_lo": -0.004756770202844495,
  "primary_hi": 0.04240067864351982,
  "secondary_mean": 0.031839999999999986,
  "secondary_lo": 0.0,
  "secondary_hi": 0.064,
  "ceiling": 0.07049103830653236,
  "verdict": "FAIL"
}
```

## Trap checks

- GREEN `oracle_leakage` — forbidden_features=[]
- GREEN `exact_token_counts` — token_cost_source='simplescaling/s1-32B tokenizer (exact)'
- GREEN `split_leakage` — selection=250 evaluation=250 overlap=0
- GREEN `secret_scan` — files_with_credentials=[]

## Notes

No probe. The probe must earn its own token cost and that is a separate question.
