# E0031 — Preregistration: does benchmark difficulty enable useful allocation?

> **REVISED before execution, on synthetic evidence only.** The first draft
> optimised `FRAC` on calibration. A power analysis (`scripts/e0031_power.py`,
> no real data) showed that optimising it against the fixed baseline drives it
> to 1.0 — where both arms take every eligible problem, the ranking is never
> consulted, and governor-minus-random is **zero by construction**. Discordant
> pairs numbered ~5 of 225 and the criterion was unreachable at ANY effect size.
> `FRAC` is now a **design constant of 0.5**, which maximises the contrast
> (~81 expected discordant pairs). Only `K_EXTRA` is fitted. See §9.

**Written and committed BEFORE any fitting.** The evaluation set has never been
scored by any analysis in this project.

| | |
|---|---|
| Predecessor | E0029-QWEN-corrected (INCONCLUSIVE-POST-HOC) |
| Feature admissibility | E0030, `difficulty` = `CONTEST_RATING_DERIVED`, admissible |
| Dataset | `livecodebench/code_generation_lite`, snapshot `0fe84c3912ea0c4d4a78037083943e8f0c4dd505` |
| Calibration | 250 problems, sha256(question_id) parity even |
| Evaluation | 225 problems, sha256(question_id) parity odd — **untouched** |
| Model | Qwen3-1.7B, 10 samples/problem, already generated and graded |

Nothing is regenerated. Nothing is re-graded. This is an allocation experiment
over data that already exists.

---

## 1. The one question

> Does the benchmark-provided `difficulty` label enable a controller to allocate
> extra samples better than spending the same tokens uniformly?

Not "is there signal" — E0029's calibration already suggested there is. The
question is whether a rule fixed in advance **transfers to problems it was not
chosen on**.

## 2. Why difficulty and nothing else

E0030 established `difficulty` is fixed at contest publication from platform
labels and setter point values, with no dependence on model performance, hidden
tests, or post-hoc evaluation. It is therefore observable **before the first
token is spent** — unlike `code_lines`, which costs a full generation to see.

`require_admissible(["difficulty"])` is asserted at runtime. The run aborts if
the provenance manifest ever reclassifies it.

## 3. The rule, fixed now

**Eligibility.** A problem is eligible for extra spend only if sample 1 failed
its private tests. Problems already solved have nothing to buy.

**Marginal-value estimate.** On calibration only, estimate

    v(d) = P(some later sample succeeds | difficulty = d, sample 1 failed)

for `d` in {easy, medium, hard}. Three numbers. No model, no fitting beyond
these three rates.

**Ranking.** Eligible problems are ordered by `v(difficulty)` descending. Ties
within a difficulty level are broken by `sha256(question_id)`, which is
deterministic, arbitrary, and independent of every outcome. The tiebreak is
specified here because a 3-level feature is mostly ties, and leaving it
unspecified would let the ordering be chosen after seeing results.

**Allocation.** The top `FRAC` of eligible problems receive `K_EXTRA` further
samples. All other problems stop at sample 1.

**Budget, fixed by design.** `FRAC = 0.5`. Half the eligible problems receive
extra samples. This is NOT fitted. Optimising it against the fixed baseline
drives it to 1.0, where every arm makes the identical allocation and the
experiment cannot answer its own question. Fixing it also makes the question the
right one: *given a fixed extra-compute budget, does difficulty tell you where to
spend it?* — rather than *how much should you spend?*, which is a different
question with an obvious answer.

**Free parameter.** `K_EXTRA` only, chosen ONCE on calibration, written to
`results/E0031_frozen.json` and **committed** before evaluation runs.

## 4. Comparisons, all at matched realised cost

| arm | definition |
|---|---|
| **Governor** | the rule above |
| **best-fixed** | k samples for every problem, k interpolated to match the Governor's realised mean token cost |
| **random** | same `FRAC` and `K_EXTRA`, subset chosen at random — isolates whether the *ranking* does anything, as opposed to the spending shape |
| **myopic** | same `FRAC` and `K_EXTRA`, ranked by `best_pub_frac` descending — what a controller would do with public-test evidence and no difficulty label |
| **oracle** | same `FRAC` and `K_EXTRA`, ranked by the true label — the observable ceiling for this spending shape |

`random` is the control that matters. An allocator that beats fixed spend but
not a random subset of the same size has discovered the *shape* of the schedule,
not a signal.

## 5. Primary outcome

    advantage = U_governor − U_best-fixed(at the Governor's realised mean cost)

`U` = fraction of problems whose private tests all pass. Cost = `total_tokens`,
exact tokenizer counts. 95% CI by **paired** bootstrap over problems, 4000
resamples, pairing on problem id.

**Success:** 95% CI excludes zero **and** the Governor also beats `random` with
a paired CI excluding zero.

Both conditions are required. Beating fixed spend alone is satisfiable by
spending shape and would not answer the question.

**Failure:** either condition unmet. Reported as a negative result.

## 6. Fixed in advance

* One evaluation pass. No reruns, no re-tuned operating point, no second feature.
* All 15 trap checks must be GREEN. Any RED blocks the result regardless of the
  numbers.
* `frozen_before_heldout` must be GREEN, meaning the freeze artifact was
  committed at an earlier commit than the evaluation run.
* Ceiling for reference: **+0.1378** (evaluation, already measured, label-only).

## 9. Power, measured before spending anything

`scripts/e0031_power.py`, 300 replicates, synthetic problems with a known
difficulty effect. `spread` scales the effect; **spread ≈ 1.0 is roughly what
E0029's calibration suggested**.

| spread | v(easy) | v(hard) | discordant | boot n=225 | McNemar n=225 | cross-fit n=475 |
|---|---|---|---|---|---|---|
| 0.00 | 0.210 | 0.210 | 17 | 1.7% | 2.7% | 5.7% |
| 0.75 | 0.352 | 0.068 | 17 | 34.0% | 41.0% | 72.7% |
| 1.00 | 0.399 | 0.021 | 17 | 55.0% | 59.7% | **92.0%** |
| 1.50 | 0.493 | 0.004 | 19 | 77.7% | **83.0%** | 98.7% |
| 2.00 | 0.588 | 0.004 | 22 | **88.7%** | 94.3% | 100% |

Row one is the false-positive rate; all three sit at or below nominal.

**The held-out design at n=225 reaches 80% power only at spread ≥ 1.5** — a
stronger effect than calibration suggests. It is therefore likely to return an
ambiguous null. This is stated before the result, not after.

## 7. Predicted outcome, recorded so it can be wrong

I expect **the CI to include zero.** Calibration gives 34 positives across three
difficulty levels; the power analysis for effects of this size indicated roughly
370 problems are needed, and evaluation supplies 162 decision points. E0031 is
underpowered by construction and this is stated before the result, not after.

The informative outcomes are:

* **Governor beats fixed AND random** — the first clean out-of-sample evidence
  in this project that a legitimately observable signal allocates compute.
* **Governor beats fixed but NOT random** — the gain is the spending shape,
  not the signal. That would retire the difficulty hypothesis and redirect
  effort to schedule design.
* **Neither** — with n=162 this is weak evidence of absence, not evidence of
  absence. The honest conclusion would be that the design needs more problems,
  and that is a data-collection decision, not an analysis one.

## 8. Known weaknesses

**Coarse feature.** Three levels means the ranking is mostly ties, resolved by
an arbitrary hash. The rule is closer to "spend on easy problems first" than to
a ranking, and it cannot discriminate within a level.

**CodeForces ratings** are computed from human contestant performance after the
contest — human, not model, independent of the hidden tests, affecting 9 of 475
problems. Recorded in the E0030 audit.

**Deployment.** A controller reading `difficulty` needs that metadata at
decision time. True for benchmark evaluation; not guaranteed for a novel
problem. Any generalisation claim is contingent on it.

**Single model, single benchmark.** Qwen3-1.7B on LiveCodeBench.

---

Frozen at commit: recorded in `results/E0031_frozen.json` at freeze time.
