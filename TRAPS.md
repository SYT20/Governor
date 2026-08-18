# Trap inventory

Fifteen executable checks. Each exists because of a specific failure that had
already printed a plausible result. **A red trap forces `verdict="BLOCKED"` in
the ledger regardless of what the caller passes** — the caller does not get to
grade its own work. **Missing evidence is red**, so silence cannot read as
success.

Code: `governor/harness/traps.py`. Registration: `run_trap_checks`.

| # | trap | failure mode it catches | motivated by | blocks PASS |
|---|---|---|---|---|
| 1 | `greedy_collapse` | controller has silently become greedy: identical calls, Δ=0 | Env 5 scored Δ=+0.0000 with 2.00 calls/episode | yes |
| 2 | `constant_schedule` | allocation constant across observable states — a schedule, not a controller | Env 5's "never spend early" beat everything | yes |
| 3 | `oracle_leakage` | a forbidden name in the feature list | Env 4a read `cfg.sigma_other` while the docstring denied it | yes |
| 4 | `answered_vs_utility` | `answered == utility` exactly ⇒ provider failures, not model quality | Gemini: 492/500 HTTP 429 | yes |
| 5 | `token_accounting` | nominal cost charged instead of measured | MathM2 charged 1.0; an LLM must charge what the runtime reports | yes |
| 6 | `execution_vs_scoring` | scoring and execution are separate code paths | Run I scored policies off one trajectory it never executed | yes |
| 7 | `progress_as_cognition` | a progress counter filed as cognitive state | `n_blocks_touched` manufactured +0.035 | yes |
| 8 | `mc_convergence` | "signal" that is unremoved estimator noise | CUBE-NM: n_mc=32 inverted the conclusion | yes (conditional) |
| 9 | `invariant_as_intelligence` | decisions constant within every cell — a lookup table | the any-time switch was a 6-entry lookup | yes |
| 10 | `frozen_before_heldout` | selection rules not frozen before the held-out run | Env 5 chose a config because its aggregate sat near zero | yes |
| 11 | `split_leakage` | the set a config was chosen on overlaps the set it is scored on | Phase 4R structural search | yes |
| 12 | `secret_scan` | credentials in tracked sources | a key was committed once | yes |
| 13 | `exact_token_counts` | estimated token costs | `len/4` was off 25–35% and the error moved with the budget level | yes |
| 14 | `budget_adherence` | a policy compared at a budget it exceeded | **E0019 spent 973 against 846 and "won" by +0.0282** | yes |
| 15 | `withdrawn_result_promotion` | a withdrawn result cited as current evidence | the ledger is append-only, so E0019 still reads PASS on disk | yes |

## Which of these actually fired

Not hypothetical. In order:

- **1, 2, 9 together** — on a degenerate smoke fixture. The traps were right and the fixture was wrong.
- **10** — on its own self-comparison bug (it compared the run's commit to itself).
- **3** — on `difficulty_ord`, a human contest label, in the LiveCodeBench adapter.
- **14** — twice: the E0019 withdrawal, and a 3.8% GPQA overrun caused by a reserve that assumed `cost ≤ level` (true on MATH, false on GPQA).
- **13, 14, 15** — each reddened a previously-green pipeline test the moment it was added, which is the check working on first contact.
- **15** — enforced by `tests/test_withdrawal.py`, which scans `FINAL-CLAIMS.md`.

## Design rules

1. **A red trap forces BLOCKED.** `ExperimentRun.finalize` overrides the caller's verdict. E0025 was recorded `BLOCKED` after I passed `INCONCLUSIVE`.
2. **Missing evidence is red.** Every check whose inputs are absent returns a failure. `secret_scan` is the sole exemption — it needs no evidence.
3. **Never remove a trap to unblock a result.** Diagnose, fix, add a regression test, rerun.
4. **Traps are checked against the import graph or the data, not the prose.** Two versions of `oracle_leakage`-style checks grepped source text and failed on their own docstrings.
