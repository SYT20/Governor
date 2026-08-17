# E0006-ceiling-gate — Phase 4R ceiling gate on the S1+S2-selected configuration

**Verdict: GATE-INCONCLUSIVE-NEED-ITEMS**

| field | value |
|---|---|
| model | `qwen/qwen3.6-27b` |
| commit | `bb1c413df1898c1ce5dad1785765f07308aa0af9` |
| budget | `{"episode_total_tokens": 2868, "low": 300, "high": 700, "n_items": 6, "charged": "usage.total_tokens"}` |
| seeds | `{"pool": 1000, "grouping": 7, "bootstrap": 0}` |
| split | `{"in_selection_items": 40, "held_out_items": 0, "independent_unit": "ITEM (episodes are groupings of a shared pool, so an episode-level CI would be anticonservative)"}` |
| raw rows | 2 (`raw.jsonl`, sha256 `1f85e840726178bb`) |
| wall | 1.1 s |

## Metric

U(clairvoyant optimum) - U(budget-limited greedy) through the canonical executor; 95% CI from a cluster bootstrap over items; PASS iff the CI lower bound exceeds 0.02

## Summary

```json
{
  "verdict": "GATE-INCONCLUSIVE-NEED-ITEMS",
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
  "held_out": null
}
```

## Trap checks

- GREEN `secret_scan` — files_with_credentials=[]

## Notes

No API calls. Configuration chosen by S1+S2 in E0005 before any ceiling was consulted.
