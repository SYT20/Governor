# E0028-marginal-ranker — Learned marginal-value ranking with trajectory and static features

**Verdict: INCONCLUSIVE**

| field | value |
|---|---|
| model | `Gemini-Pro-1.5 (May) generations, published by LiveCodeBench` |
| commit | `a46315f643681f5a407333ee4f744164d48ae9c3` |
| budget | `{"axis": "rank-then-allocate; sample 1 everywhere, remainder to top-ranked", "B_star": 240.6580310880829, "charged": "exact tokenizer count over published generations"}` |
| seeds | `{"split": "sha256(question_id) parity", "cv": 0, "bootstrap": 0}` |
| split | `{"calibration": 207, "evaluation": 193}` |
| raw rows | 193 (`raw.jsonl`, sha256 `efe4b2eb6d6f9640`) |
| wall | 0.2 s |

## Metric

pass-within-k at matched realised cost; primary = ranked policy minus the randomised fixed envelope at its own cost

## Summary

```json
{
  "governor_U": 0.37305699481865284,
  "fixed_matched": 0.3660229988347621,
  "primary_mean": 0.007033995983890762,
  "primary_lo": -0.01276506820032967,
  "primary_hi": 0.028685709001742863,
  "oracle_same_frac": 0.050543326612207506,
  "heldout_auc": 0.6300008098477486,
  "heldout_precision_at_10": 0.08490566037735849,
  "heldout_lift_at_10": 1.8315363881401616,
  "heldout_ndcg": 0.5173662752110926,
  "cv_auc_old_features": 0.5827129593975955,
  "cv_auc_rich_features": 0.6467077396996534,
  "calibration_positives": 49,
  "decision_rows": 1003,
  "events_per_feature": 1.96,
  "paired_sd": 0.15033389027960303,
  "n_required_for_eps_0.02": 217.05307535362377,
  "frac": 0.1,
  "model": "rf",
  "verdict": "INCONCLUSIVE"
}
```

## Trap checks

- GREEN `greedy_collapse` — mean_delta=+0.007034 calls_identical=False
- GREEN `constant_schedule` — distinct_decision_patterns=193
- GREEN `oracle_leakage` — forbidden_features=[]
- GREEN `answered_vs_utility` — answered=1.0000 utility=0.3731 -> ok
- GREEN `token_accounting` — charged==used:True over_budget_calls=0
- GREEN `execution_vs_scoring` — scored_through_canonical_executor=True
- GREEN `progress_as_cognition` — unjustified_progress_features=[]
- GREEN `invariant_as_intelligence` — cells_with_constant_decision=1/3
- GREEN `frozen_before_heldout` — froze=b8e2884 heldout=HEAD
- GREEN `split_leakage` — selection=207 evaluation=193 overlap=0
- GREEN `exact_token_counts` — token_cost_source='exact tokenizer count over published LiveCodeBench generations'
- GREEN `budget_adherence` — realised=241 budget=241 over=+0.0%
- GREEN `withdrawn_result_promotion` — cited=2 withdrawn=2 promoted=[]
- GREEN `secret_scan` — files_with_credentials=[]

## Notes

Operating point frozen on calibration before a single evaluation application. E0027's rank-fraction sweep touched evaluation and was reported as diagnostic; this does not.
