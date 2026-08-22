# E0029-QWEN-corrected — INCONCLUSIVE (post-hoc)

Re-analysis of the **same** 4750 generation rows and 4750 private-test grades
after the static-feature defect was fixed. Nothing was regenerated and nothing
was re-graded.

**This is post-hoc.** The defect was found after the preregistered run reported
its outcome, so this corrects an engineering error; it does not convert the run
into a preregistered test.

## What changed

Constant features at the allocation point fell from **17/25 to 7/25**. The
remaining seven are genuinely degenerate with a single attempt (`attempt_idx`,
`attempts_left`, `pub_frac_trend`, `code_len_var`, `n_distinct_outcomes`,
`all_same_pub_frac`, `n_try`).

## What it shows

The signal is program **size**, and it is not in the public-test outcome at all:

| feature | AUC |
|---|---|
| `code_lines` | 0.651 |
| `ast_call_nodes` | 0.647 |
| `best_pub_frac` — what the original run rested on | **0.551** |

Best-of-18-features null: 0.646. Margin **+0.005**, permutation p **0.041**.

Fewer features win monotonically — `code_lines` alone (0.647) beats all 25
(0.557). With 34 positives, extra features are variance, not evidence.

## The confound that is not cleared

Difficulty. Rescue rate spans **6.3x** across strata and code length tracks it
exactly. `difficulty` alone scores **0.707** — better than `code_lines`, and free
to observe. Within difficulty strata `code_lines` falls to 0.567.

## Status

Gate 3 still fails. The **225 evaluation problems remain untouched** and are
available for exactly one confirmatory test.
