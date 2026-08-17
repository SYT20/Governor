# E0010-phase5-design — Phase 5 design: configurations screened by the derived headroom law

**Verdict: CONFIG-FOUND**

| field | value |
|---|---|
| model | `qwen/qwen3.6-27b` |
| commit | `aab0e2d3300708da86fe442988a12a17ad90c31d` |
| budget | `{"swept": "per configuration"}` |
| seeds | `{"pool": 1000, "grouping": 7}` |
| split | `{"frozen_by": "configs/phase4r_split.json", "selection_items": 40}` |
| raw rows | 146 (`raw.jsonl`, sha256 `06f0012a73b715f4`) |
| wall | 20.5 s |

## Metric

S1 = ceiling(n,k,p) from the closed form; S2 = actual/cap; S3 = ceiling(n,k-1)/ceiling(n,k) and cross-split k drift; measured ceiling = U(clairvoyant) - U(greedy) via the executor

## Summary

```json
{
  "verdict": "CONFIG-FOUND",
  "n_configs": 146,
  "n_passing_all": 10,
  "chosen": {
    "n_items": 12,
    "low": 300,
    "high": 700,
    "budget": 6486,
    "n_sel_items": 40,
    "n_ev_items": 55,
    "p": 0.475,
    "k_sel": 5.666666666666667,
    "k_ev": 5.75,
    "k": 6,
    "ideal": 0.19275932722944855,
    "ideal_at_k_minus_1": 0.1870329130534857,
    "s3_drop": 0.029707585403358827,
    "act_over_cap": 0.8030797101449276,
    "measured_selection": 0.25,
    "realised": 1.2969541012270487,
    "S1": true,
    "S2": true,
    "S3": true
  }
}
```

## Trap checks

- GREEN `secret_scan` — files_with_credentials=[]
- GREEN `split_leakage` — selection=40 evaluation=360 overlap=0

## Notes

Selection by S1/S2/S3 only. The measured ceiling is reported for every cell and never used to select.
