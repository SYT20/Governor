# E0019-predictor-loss-math — Predictor loss and calibration, no probe (math)

**Verdict: PASS**

| field | value |
|---|---|
| model | `s1-32B via simplescaling/results` |
| commit | `4f5e1ca8a02dafc6ba5235d37a282708e3303a0c` |
| budget | `{"contract": "SOFT_EXPECTED_BUDGET", "reported_at": 845.7475505084745, "charged": "simplescaling/s1-32B tokenizer (exact)"}` |
| seeds | `{"split": "doc_id parity", "bootstrap": 0}` |
| split | `{"calibration_items": 250, "evaluation_items": 250}` |
| raw rows | 28 (`raw.jsonl`, sha256 `c5687a6b7560972b`) |
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
  "primary_mean": 0.028175873024499625,
  "primary_lo": 0.0032500402145681445,
  "primary_hi": 0.052476586178732826,
  "secondary_mean": 0.03641599999999998,
  "secondary_lo": -0.002099999999999996,
  "secondary_hi": 0.08400000000000007,
  "ceiling": 0.16379740226795358,
  "verdict": "PASS"
}
```

## Trap checks

- GREEN `oracle_leakage` — forbidden_features=[]
- GREEN `exact_token_counts` — token_cost_source='simplescaling/s1-32B tokenizer (exact)'
- GREEN `split_leakage` — selection=250 evaluation=250 overlap=0
- GREEN `secret_scan` — files_with_credentials=[]

## Notes

No probe. The probe must earn its own token cost and that is a separate question.
