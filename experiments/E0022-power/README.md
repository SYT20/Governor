# E0022-power — Power analysis for the real-LLM claim

**Verdict: SAMPLE-UNATTAINABLE**

| field | value |
|---|---|
| model | `s1-32B via simplescaling/results` |
| commit | `6160560108a89f704894e4b118fc0dc398ec0eca` |
| budget | `{"B_star": 846.0, "charged": "simplescaling/s1-32B tokenizer (exact)"}` |
| seeds | `{"split": "doc_id parity"}` |
| split | `{"evaluation_items": 250, "benchmark_total": 500}` |
| raw rows | 2 (`raw.jsonl`, sha256 `2d1230940800e8cb`) |
| wall | 0.3 s |

## Metric

items required for a 95% CI lower bound above zero, from the observed per-item paired mean and standard deviation

## Summary

```json
{
  "n_required_all_items": 26031.082730923696,
  "n_required_disagreements": 11199.206279245283,
  "benchmark_size": 500,
  "multiple_of_benchmark": 52.06216546184739,
  "disagreement_rate": 0.428,
  "verdict": "SAMPLE-UNATTAINABLE"
}
```

## Trap checks

- GREEN `exact_token_counts` — token_cost_source='simplescaling/s1-32B tokenizer (exact)'
- GREEN `secret_scan` — files_with_credentials=[]

## Notes

Corrects a previous 2000-3000 item estimate that scaled from a CI half-width instead of per-item variance.
