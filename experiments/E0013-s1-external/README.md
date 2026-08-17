# E0013-s1-external — External replication on s1 released generations (MATH-500)

**Verdict: EXTERNAL-S2-FAILS**

| field | value |
|---|---|
| model | `s1-32B via simplescaling/results` |
| commit | `5e409f274cbd9d63bf792f4b9e2ad85f0c6ccbbe` |
| budget | `{"levels": [500, 1000, 2000, 4000, 8000, 16000, 32000], "charged": "generation tokens", "reservation": "hard worst-case cap"}` |
| seeds | `{"split": "doc_id parity"}` |
| split | `{"selection_items": 250, "evaluation_items": 250, "note": "third-party data; no item, model, prompt or budget chosen by us"}` |
| raw rows | 28 (`raw.jsonl`, sha256 `ef8f4e37cb1e610c`) |
| wall | 1.4 s |

## Metric

S1 = ceiling(12,k,p) from the closed form; S2 = mean actual generation tokens / (prompt + cap)

## Summary

```json
{
  "verdict": "EXTERNAL-S2-FAILS",
  "n_pairs": 21,
  "n_pass_S1": 5,
  "n_pass_both": 0,
  "best_ideal": 0.16801911169619121,
  "best_act_over_cap": 0.6808496240601504
}
```

## Trap checks

- GREEN `secret_scan` — files_with_credentials=[]

## Notes

No API calls. https://huggingface.co/datasets/simplescaling/results
