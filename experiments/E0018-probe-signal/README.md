# E0018-probe-signal — Early-reasoning probe features: signal check and E0017 retraction

**Verdict: PROBE-SIGNAL-ON-MATH**

| field | value |
|---|---|
| model | `s1-32B via simplescaling/results` |
| commit | `8817befd98786b41bd1ce464d7f8fb5967767e83` |
| budget | `{"probe_tokens": 500, "charged": "simplescaling/s1-32B tokenizer (exact)"}` |
| seeds | `{"split": "doc_id parity", "cv": 0}` |
| split | `{"rule": "doc_id parity; calibration half only"}` |
| raw rows | 8 (`raw.jsonl`, sha256 `83fdbc61d76d62c8`) |
| wall | 0.3 s |

## Metric

cross-validated AUC for P(gain>0) from question-only, probe-only and combined features; ridge R2 reported alongside to show the metric artefact

## Summary

```json
{
  "verdict": "PROBE-SIGNAL-ON-MATH",
  "best_auc_math": 0.7408088235294118,
  "best_auc_gpqa": 0.5416666666666667,
  "retracts": "E0017 diagnosis (not its result)"
}
```

## Trap checks

- GREEN `oracle_leakage` — forbidden_features=[]
- GREEN `exact_token_counts` — token_cost_source='simplescaling/s1-32B tokenizer (exact)'
- GREEN `split_leakage` — selection=1 evaluation=1 overlap=0
- GREEN `secret_scan` — files_with_credentials=[]

## Notes

Answer stability across budgets is deliberately excluded: it reads the outcome the allocator is trying to predict.
