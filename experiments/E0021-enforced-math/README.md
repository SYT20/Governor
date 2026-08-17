# E0021-enforced-math — Budget-enforced Governor at matched realised cost (math)

**Verdict: INCONCLUSIVE**

| field | value |
|---|---|
| model | `s1-32B via simplescaling/results` |
| commit | `b3ce7df3714daa82bd26be5110e30abd4cfd549b` |
| budget | `{"contract": "SOFT_EXPECTED_BUDGET + hard runtime cap", "B_star": 845.7475505084745, "charged": "simplescaling/s1-32B tokenizer (exact)"}` |
| seeds | `{"split": "doc_id parity", "bootstrap": 0}` |
| split | `{"calibration_items": 250, "evaluation_items": 250}` |
| raw rows | 250 (`raw.jsonl`, sha256 `97cf04975f2ea66a`) |
| wall | 0.2 s |

## Metric

primary = Governor minus the fixed envelope AT THE GOVERNOR'S OWN REALISED COST; secondary = Governor minus myopic; 95% CI from a bootstrap over evaluation items with the controller frozen

## Summary

```json
{
  "benchmark": "math",
  "B_star": 845.7475505084745,
  "governor_U": 0.788,
  "governor_tokens": 845.0,
  "unenforced_tokens": 973.484,
  "fixed_at_matched_cost": 0.7839630154876651,
  "primary_mean": 0.004299469140605342,
  "primary_lo": -0.04877645931250004,
  "primary_hi": 0.04337600430865811,
  "secondary_mean": 0.012410000000000003,
  "secondary_lo": -0.028000000000000025,
  "secondary_hi": 0.05599999999999994,
  "mcnemar_p": 0.7011080384254456,
  "n_disagree": 102,
  "verdict": "INCONCLUSIVE"
}
```

## Trap checks

- GREEN `oracle_leakage` — forbidden_features=[]
- GREEN `exact_token_counts` — token_cost_source='simplescaling/s1-32B tokenizer (exact)'
- GREEN `split_leakage` — selection=250 evaluation=250 overlap=0
- GREEN `budget_adherence` — realised=845 budget=846 over=-0.1% baseline=845 baseline_short_by=+0.0%
- GREEN `secret_scan` — files_with_credentials=[]

## Notes

E0019 is withdrawn: it spent 15% over budget and was scored against a baseline at the nominal budget.
