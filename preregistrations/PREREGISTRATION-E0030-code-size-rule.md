# E0030 — Preregistration: program size as the allocation signal

**Status:** WRITTEN, NOT EXECUTABLE. Blocked pending `scripts/e0030_validate.py`.

> **Do not run the evaluation on the strength of this document as first drafted.**
> §1 cited "null for best of 6 models: 0.625" as though `code_lines` had cleared
> it. The relevant comparison is against the best-of-18-**features** null of
> 0.646, which `code_lines` exceeds by **+0.001** — and the model search was a
> further look on top of the feature search, so the honest threshold is higher
> again. A +0.001 margin is not a finding, and spending the only untouched
> confirmatory set on it would convert it into a coin flip.
>
> Four confounds must clear first: the generation cap (long == truncated ==
> failed by construction), problem difficulty, the joint search null, and
> whether the ranking buys any utility at all. If T3 returns p >= 0.05 this
> preregistration is withdrawn rather than executed.
**Predecessor:** E0029-QWEN, which stopped at gate 3 and remains INCONCLUSIVE.
**Evaluation set:** 225 LiveCodeBench problems, sha256(question_id) parity odd.
Never scored by any analysis to date.

---

## 1. Where this comes from

E0029 generated 4750 Qwen3-1.7B samples over 475 problems and graded them
against LiveCodeBench's private tests. It stopped at gate 3: the 25-feature
ranker scored AUC 0.557 at the allocation point against a permutation null of
0.597. That result stands and is not superseded by this document.

Diagnostics afterwards — on calibration only — found two things.

**The feature vector was broken for the whole of E0029.** `decision_features`
read the eleven static/AST features as precomputed columns with a default of
0.0. E0029's generation rows carry `code` and no such columns, so every static
feature was zero for every sample. E0029 therefore tested five distinct
observables, not the twenty-five it was designed around. Fixed in `d23554e`.

**With those features alive, the signal is program size, and it is not in the
public-test outcome at all.** Calibration, n=162, 34 positives:

| feature | AUC | direction |
|---|---|---|
| `code_lines` | 0.651 | lower is better |
| `ast_call_nodes` | 0.647 | lower |
| `code_chars` | 0.645 | lower |
| `ast_nodes` | 0.645 | lower |
| `has_recursion` | 0.623 | lower |
| `best_pub_frac` | **0.551** | higher |

Null for the best of 18 live features: **0.646**. `code_lines` sits at 0.647 —
a margin of **+0.001**, which is not evidence of anything. `best_pub_frac`, the
feature E0029 effectively rested on, is worth almost nothing either.

**More features make it worse, monotonically:**

| model | features | allocation-point AUC |
|---|---|---|
| `code_lines` alone | 1 | **0.647** |
| logistic-l1 | 25 | 0.559 |
| logistic | 25 | 0.557 |
| logistic, size family only | 7 | 0.540 |

Null for the best of 6 models: 0.625 — but that search RE-USES the feature
already chosen from 18, so it is not an independent confirmation and the two
thresholds cannot be quoted side by side as though they were.
With 34 positives this is what one expects: the size measures correlate hard, so
seven of them is not seven times the evidence, it is one signal plus six ways to
overfit.

## 2. Hypothesis

> Among problems whose first sample failed, **shorter first attempts are more
> likely to be rescued by resampling.** An allocator that spends its extra
> samples on the shortest-first-attempt problems will beat a fixed allocation of
> equal token cost.

Mechanism, stated so it can be wrong: when Qwen emits a short program it has
found a compact idea, and the failure is likely a near miss that a different
sample repairs. A long, sprawling program indicates the model has not found the
idea at all, and further samples explore the same wrong neighbourhood.

## 3. The rule, fixed now

Score each problem by `code_lines` of **sample 1 only** — ascending, so shortest
first. No model, no fitting, no other feature.

    score(problem) = -code_lines(sample_1)

Problems solved by sample 1 are excluded from spending: nothing to buy.

Allocation: the top `FRAC` of remaining problems by score each receive `K_EXTRA`
further samples. Everyone else stops at sample 1.

`FRAC` and `K_EXTRA` are chosen ONCE on **calibration only**, by maximising
advantage over the matched-cost fixed baseline, then frozen to
`results/E0030_frozen.json` and committed before evaluation runs.

## 4. Primary outcome

    advantage = U_governor − U_fixed(at the same mean token cost)

`U` is the fraction of problems solved, where solved means **all private tests
pass**. The baseline is the best fixed "k samples for everyone", linearly
interpolated to exactly match the Governor's realised mean cost. 95% CI by
bootstrap over problems, 4000 resamples.

**Success:** the 95% CI excludes zero.
**Failure:** it does not. That is a publishable result and will be reported.

There is one evaluation. No reruns, no adjusted operating point, no second
feature. If this fails, E0030 is negative and the next experiment needs new
data, not new analysis.

## 5. Fixed in advance

| | |
|---|---|
| Evaluation problems | 225, sha256 parity odd |
| Calibration problems | 250, sha256 parity even |
| Feature | `code_lines` of sample 1, ascending |
| Label | `hidden_all_passed` from private tests |
| Cost | `total_tokens`, exact tokenizer counts |
| Baseline | best fixed-k, cost-matched by interpolation |
| CI | 4000-resample bootstrap over problems |
| Ceiling | +0.1378 (evaluation, already measured) |
| Traps | all 15 must be GREEN; any RED blocks |

## 6. What would falsify the mechanism

* Advantage CI includes zero → the rule does not transfer.
* Advantage is negative → size is anti-predictive out of sample, and the
  calibration result was noise dressed as signal at 34 positives.
* `frozen_before_heldout` RED → the protocol was not followed and no number
  from the run may be reported, whatever it says.

## 7. Known weaknesses, stated before the result

**The feature was chosen after seeing calibration.** That is what calibration is
for, and evaluation is untouched, so a single test is valid. But the honest
description of E0030 is "one confirmatory test of a hypothesis generated by
E0029's calibration set", not an independent replication.

**The effect is small and the sample is thin.** Calibration AUC 0.647 with 34
positives has a standard error near 0.08. The power calculation says an effect
this size needs roughly 370 problems to resolve at 80% power; evaluation offers
162 decision points. **E0030 is underpowered by construction**, and a null result
will be genuinely ambiguous between "no effect" and "too few problems". A
positive result under these conditions is the informative outcome; a negative one
is weak evidence.

**Single model, single benchmark.** Qwen3-1.7B on LiveCodeBench. Nothing here
claims the size heuristic transfers to other models or task families, and the
mechanism above suggests it might not for models that do not fail this way.

---

Frozen at commit: recorded in `results/E0030_frozen.json` at freeze time.
