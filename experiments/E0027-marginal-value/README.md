# E0027-marginal-value — Where the execution-feedback signal fails to become utility

**Verdict: INCONCLUSIVE**

| field | value |
|---|---|
| model | `Gemini-Pro-1.5 (May) generations, published by LiveCodeBench` |
| commit | `c14df207caad05aa011dfe1582338acbad6cc654` |
| budget | `{"axis": "rank-then-allocate; sample 1 everywhere, remainder to top-ranked", "B_star": 1002.4093264248704, "charged": "exact tokenizer count over published generations"}` |
| seeds | `{"split": "sha256(question_id) parity", "bootstrap": 0}` |
| split | `{"calibration": 207, "evaluation": 193}` |
| raw rows | 193 (`raw.jsonl`, sha256 `1d481a8f5400e66f`) |
| wall | 0.1 s |

## Metric

pass-within-k at matched realised cost; primary = ranked policy minus the randomised fixed envelope at its own cost

## Summary

```json
{
  "observable_ceiling": 0.0588,
  "oracle_ranking_adv": 0.050543326612207506,
  "oracle_ranking_lo": 0.020302006970908535,
  "oracle_ranking_hi": 0.08163140951376187,
  "primary_mean": 0.005528507431274476,
  "primary_lo": -0.010362694300518135,
  "primary_hi": 0.021775548451107987,
  "random_adv": 0.006236858720598093,
  "marginal_auc": 0.6316758747697975,
  "positives_calibration": 19,
  "positives_evaluation": 12,
  "events_per_feature": 2.7142857142857144,
  "breakeven_purity": 0.9378238341968912,
  "achieved_purity": 0.917,
  "cost_ratio": 16.083333333333336,
  "tokens_per_utility_point": 19426.916666666668,
  "verdict": "INCONCLUSIVE"
}
```

## Trap checks

- GREEN `greedy_collapse` — mean_delta=+0.005529 calls_identical=False
- GREEN `constant_schedule` — distinct_decision_patterns=117
- GREEN `oracle_leakage` — forbidden_features=[]
- GREEN `answered_vs_utility` — answered=1.0000 utility=0.4145 -> ok
- GREEN `token_accounting` — charged==used:True over_budget_calls=0
- GREEN `execution_vs_scoring` — scored_through_canonical_executor=True
- GREEN `progress_as_cognition` — unjustified_progress_features=[]
- GREEN `invariant_as_intelligence` — cells_with_constant_decision=0/3
- GREEN `frozen_before_heldout` — froze=b8e2884 heldout=HEAD
- GREEN `split_leakage` — selection=207 evaluation=193 overlap=0
- GREEN `exact_token_counts` — token_cost_source='exact tokenizer count over published LiveCodeBench generations'
- GREEN `budget_adherence` — realised=1002 budget=1002 over=+0.0%
- GREEN `withdrawn_result_promotion` — cited=2 withdrawn=2 promoted=[]
- GREEN `secret_scan` — files_with_credentials=[]

## Notes

Corrects E0026's +0.1880 ceiling, which used an oracle deciding before paying for sample 1. Observable ceiling is +0.0588.
