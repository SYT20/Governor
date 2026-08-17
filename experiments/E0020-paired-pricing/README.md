# E0020-paired-pricing — Paired Governor-vs-myopic on decision disagreements (MATH)

**Verdict: PRICING-UNRESOLVED**

| field | value |
|---|---|
| model | `s1-32B via simplescaling/results` |
| commit | `4b1368899256d67cd8807644a7794ff9e7996299` |
| budget | `{"contract": "SOFT_EXPECTED_BUDGET", "B_star": 846.0, "charged": "simplescaling/s1-32B tokenizer (exact)"}` |
| seeds | `{"split": "doc_id parity", "bootstrap": 0}` |
| split | `{"calibration_items": 250, "evaluation_items": 250, "conditioning": "disagreement set defined by frozen policies; no held-out outcome used to select items"}` |
| raw rows | 112 (`raw.jsonl`, sha256 `b09c23a57dea1eda`) |
| wall | 0.4 s |

## Metric

McNemar exact on discordant correctness among items where the two frozen policies choose different budget levels; paired mean difference with bootstrap CI reported alongside

## Summary

```json
{
  "n_evaluation": 250,
  "n_disagree": 112,
  "b_gov_right_myo_wrong": 19,
  "c_gov_wrong_myo_right": 10,
  "mcnemar_p": 0.13604594767093658,
  "paired_all_mean": 0.035752,
  "paired_all_lo": -0.004,
  "paired_all_hi": 0.076,
  "paired_dis_mean": 0.08063392857142858,
  "paired_dis_lo": -0.008928571428571428,
  "paired_dis_hi": 0.17857142857142858,
  "gov_higher_level": 91,
  "gov_lower_level": 21,
  "verdict": "PRICING-UNRESOLVED"
}
```

## Trap checks

- GREEN `oracle_leakage` — forbidden_features=[]
- GREEN `exact_token_counts` — token_cost_source='simplescaling/s1-32B tokenizer (exact)'
- GREEN `split_leakage` — selection=250 evaluation=250 overlap=0
- GREEN `secret_scan` — files_with_credentials=[]

## Notes

Nothing changed from E0019 except the analysis.
