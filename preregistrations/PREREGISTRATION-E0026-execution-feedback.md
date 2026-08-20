# Preregistration — E0026: Execution-Feedback Allocation

**Written before any E0026 number exists.** Two findings below were established
during feasibility checking and are recorded here because they constrain the
design, not because they are results.

## Why this is a genuinely new signal

E0024 produced **45 allocation disagreements and zero outcome differences**: the
controller reallocated among problems where reallocation could not change the
result. E0025 added a two-sample agreement probe, moved the estimate in the
predicted direction (−0.0028 → +0.0046), and was recorded **BLOCKED** on a +5.2%
budget violation.

Both used *static* problem text or *agreement between samples*. Neither observed
what the attempted solution actually **did when run**. Execution feedback is
qualitatively different: it describes the trajectory of a concrete attempt.

## FINDING 1 — the free version of this experiment is invalid

LiveCodeBench's published submissions carry a `metadata` field per sample. It is
not usable, and not merely because it is unfair:

| metadata | graded PASS | graded FAIL |
|---|---|---|
| empty `{}` | 1530 | 0 |
| non-empty | 0 | 2470 |

**Metadata emptiness is the label**, exactly, on all 4000 samples. A feature as
innocuous as "was metadata empty" scores AUC 1.0 and means nothing. The contents
are worse: `expected` (the correct output) and `inputs` (the failing hidden
test).

**`metadata` is therefore forbidden in its entirety** — contents, emptiness,
length, and existence. So is `private_test_cases`, and so is `graded_list`
outside of outcome scoring.

## FINDING 2 — the signal must be generated, not read

Legitimate execution feedback has to be produced by running each candidate
against the tests **a real agent can see** — `public_test_cases`, which ships as
plain JSON, unlike the zlib-encoded private set. This costs local compute and
introduces an execution-cost axis the previous experiments did not have.

## Hypothesis

> **H:** Observable execution feedback from running a cheap first candidate
> against public tests predicts whether drawing another sample will change the
> final outcome, and a controller using it beats the best fixed sample schedule
> at matched inference cost.

## The information boundary

Permitted at decision time — all obtainable by an agent running its own code:

| Feature | Source |
|---|---|
| `compile_ok` | does the candidate parse and import |
| `runtime_error` | did execution raise |
| `timeout` | did it exceed the per-test limit |
| `pub_passed`, `pub_failed`, `pub_frac` | public tests passed / failed / fraction |
| `exec_latency_s` | wall time to run public tests |
| `output_nonempty` | did it emit anything |
| `output_format_ok` | did output shape match the expected *form*, not value |

Forbidden: `metadata` in any form, `private_test_cases`, `expected` values,
failing hidden inputs, `graded_list` as a feature, `pass@1`, and any function of
a future sample.

`oracle_leakage` (trap 3) is run over the final feature-name list. Missing
evidence is red.

## Two currencies, accounted separately

The previous experiments had a single scalar budget. This one cannot:

- **`inference_cost`** — generation tokens. **The primary budget.** The
  Governor's advantage is claimed at matched inference cost, as before.
- **`execution_cost`** — wall-clock seconds running public tests. Reported
  independently, never netted against tokens.
- **`total_system_cost`** — reported as a secondary analysis. If the Governor
  wins on tokens but loses once execution is priced in, that is stated plainly.

`budget_adherence` (trap 14) applies to `inference_cost`: over-spend ≤ 2%,
under-spend permitted, baseline never given fewer tokens than the policy used.

## Order, frozen

1. **Ceiling.** E0023 recorded CEILING-PASS at +0.057 for *non-adaptive* sample
   allocation. Sequential allocation with feedback is a different decision
   problem, so its ceiling is re-measured: a sequential oracle that stops as soon
   as a passing sample exists, against the best fixed k at matched cost. If that
   ceiling is below ε = 0.02, **stop and record it** — no predictor, no
   controller.
2. **Predictor.** Does execution feedback predict *marginal value of another
   sample* — that is, `P(some later sample passes | this one failed, features)`?
   Fit on calibration split only. Report AUC, Brier, ECE.
   **The diagnostic that matters:** measured before any controller exists.
3. **Controller.** Only if 1 and 2 both clear.

## Split

Reuse the frozen LCB split unchanged: `sha256(question_id)` parity, 210
calibration / 190 evaluation. Not re-drawn, not re-tuned. `split_leakage`
(trap 12) is red on any overlap.

## Baselines

`best_fixed` (randomised between adjacent k to match expected cost) ·
`random` · `myopic` · **Governor** · sequential oracle.

Primary: `U(Governor) − U(best_fixed)` at matched realised inference cost.
Secondary: `U(Governor) − U(myopic)`.
Paired cluster bootstrap over **problems**; McNemar on discordant outcomes.

## Stop criteria — any one halts the experiment

- sequential ceiling < 0.02
- predictor AUC ≤ 0.55 on calibration
- any red trap
- inference over-spend > 2%
- execution cost exceeds inference cost by more than 10× (the signal is not worth
  its price)
- a permitted feature turns out to be unavailable at decision time

## What would refute H

If execution feedback predicts the outcome well (high AUC) but the controller
still shows no separation, then the binding constraint is not observability at
all but the **structure of the allocation problem** — the outcome is decided
before any allocation decision can act on it. That would be a stronger and more
useful negative than E0024's, and must be reported as such rather than retried
with a fourth signal.

## Execution safety

4000 untrusted model-generated programs are run. Each in a separate subprocess,
hard timeout, temporary working directory, no shell, output capped. Timeout and
evaluator configuration are recorded in provenance — LiveCodeBench's own
documentation notes results vary with timeout settings, so this is not treated as
a deterministic oracle.
