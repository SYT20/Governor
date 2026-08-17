# E0006-ceiling-gate — Phase 4R ceiling gate on the S1+S2-selected configuration

**Verdict: CEILING-FAIL**

| field | value |
|---|---|
| model | `qwen/qwen3.6-27b` |
| commit | `5a795feb569ea8b0f832b7539c4a57d726b3d103` |
| budget | `{"episode_total_tokens": 2868, "low": 300, "high": 700, "n_items": 6, "charged": "usage.total_tokens"}` |
| seeds | `{"pool": 1000, "grouping": 7, "bootstrap": 0}` |
| split | `{"in_selection_items": 40, "held_out_items": 24, "frozen_split_sha256": "fef2ff3cebdaa4b674ff7bf89ba0ac4df02c5fa701e11cc0d919ab0354425fb4", "independent_unit": "ITEM (episodes are groupings of a shared pool, so an episode-level CI would be anticonservative)"}` |
| raw rows | 2 (`raw.jsonl`, sha256 `25aee8aed2093b2d`) |
| wall | 1.1 s |

## Metric

U(clairvoyant optimum) - U(budget-limited greedy) through the canonical executor; 95% CI from a cluster bootstrap over items; PASS iff the CI lower bound exceeds 0.02

## Summary

```json
{
  "verdict": "CEILING-FAIL",
  "config": {
    "n_items": 6,
    "low": 300,
    "high": 700,
    "budget": 2868
  },
  "in_selection": {
    "cheap": 0.05555555555555555,
    "greedy": 0.13888888888888887,
    "oracle": 0.2777777777777778,
    "best_fixed": 0.16666666666666666,
    "ceiling": 0.13888888888888892,
    "ceiling_vs_fixed": 0.11111111111111113,
    "greedy_deep": 1.5,
    "n_episodes": 6,
    "boot_mean": 0.10548611111111111,
    "ci_lo": 0.02777777777777779,
    "ci_hi": 0.19444444444444445,
    "n_items": 40,
    "passes_gate": true
  },
  "held_out": {
    "cheap": 0.041666666666666664,
    "greedy": 0.16666666666666666,
    "oracle": 0.20833333333333331,
    "best_fixed": 0.16666666666666666,
    "ceiling": 0.04166666666666666,
    "ceiling_vs_fixed": 0.04166666666666666,
    "greedy_deep": 1.25,
    "n_episodes": 4,
    "boot_mean": 0.08604166666666666,
    "ci_lo": 0.0,
    "ci_hi": 0.16666666666666666,
    "n_items": 24,
    "passes_gate": false
  }
}
```

## Trap checks

- GREEN `secret_scan` — files_with_credentials=[]
- GREEN `split_leakage` — selection=40 evaluation=24 overlap=0

## Notes

No API calls. Configuration chosen by S1+S2 in E0005 before any ceiling was consulted.
