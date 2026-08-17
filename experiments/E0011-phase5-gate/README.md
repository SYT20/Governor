# E0011-phase5-gate — Phase 4R ceiling gate on the S1+S2-selected configuration

**Verdict: CEILING-PASS**

| field | value |
|---|---|
| model | `qwen/qwen3.6-27b` |
| commit | `4a717c5dc18dc5a4ea874d46fffce97168fb95ed` |
| budget | `{"episode_total_tokens": 6486, "low": 300, "high": 700, "n_items": 12, "charged": "usage.total_tokens"}` |
| seeds | `{"pool": 1000, "grouping": 7, "bootstrap": 0}` |
| split | `{"in_selection_items": 40, "held_out_items": 55, "frozen_split_sha256": "fef2ff3cebdaa4b674ff7bf89ba0ac4df02c5fa701e11cc0d919ab0354425fb4", "independent_unit": "ITEM (episodes are groupings of a shared pool, so an episode-level CI would be anticonservative)"}` |
| raw rows | 2 (`raw.jsonl`, sha256 `96eb398d10c1fea9`) |
| wall | 1.1 s |

## Metric

U(clairvoyant optimum) - U(budget-limited greedy) through the canonical executor; 95% CI from a cluster bootstrap over items; PASS iff the CI lower bound exceeds 0.02

## Summary

```json
{
  "verdict": "CEILING-PASS",
  "config": {
    "n_items": 12,
    "low": 300,
    "high": 700,
    "budget": 6486
  },
  "in_selection": null,
  "held_out": {
    "cheap": 0.041666666666666664,
    "greedy": 0.125,
    "oracle": 0.35416666666666663,
    "best_fixed": 0.125,
    "ceiling": 0.22916666666666663,
    "ceiling_vs_fixed": 0.22916666666666663,
    "greedy_deep": 5.0,
    "n_episodes": 4,
    "boot_mean": 0.17927083333333335,
    "ci_lo": 0.08333333333333337,
    "ci_hi": 0.27083333333333337,
    "n_items": 55,
    "passes_gate": true
  }
}
```

## Trap checks

- GREEN `secret_scan` — files_with_credentials=[]
- GREEN `split_leakage` — selection=40 evaluation=55 overlap=0

## Notes

No API calls. Configuration chosen by S1+S2 in E0005 before any ceiling was consulted.
