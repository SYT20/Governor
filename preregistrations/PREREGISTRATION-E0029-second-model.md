# Preregistration — E0029: second-model replication

**Written before any generation.** The problem list and the calibration/evaluation
split are frozen here, in this document and in `configs/e0029_split.json`, so
neither can be chosen after seeing an outcome.

## Why a replication rather than another model tweak

E0028 exhausted what feature and target engineering can do on the existing data:

| | |
|---|---|
| observable ceiling | +0.0588 |
| oracle marginal ranking | +0.0505 [+0.0203, +0.0816] |
| learned ranker (E0027) | +0.0055 [−0.0104, +0.0218] |
| learned ranker (E0028, richer target and features) | +0.0070 [−0.0128, +0.0287] |
| held-out AUC | 0.632 → 0.630 |

The target reframing tripled the positives and the features raised grouped CV AUC
from 0.583 to 0.647. **The decision did not move.** Another round on the same 193
evaluation problems would answer nothing.

Three explanations remain, and only a second, independently generated model
separates them:

- **A** — specific to LiveCodeBench *and* to Gemini-Pro-1.5's error distribution
- **B** — a general property of observable allocation
- **C** — an evaluation-sample limitation (E0028 needs n≈217, has 193)

## Hypothesis

> **H:** The E0028 result reproduces on a second reasoning model. Specifically,
> the learned ranker's advantage over the matched-cost fixed baseline has a
> confidence interval containing zero, while the oracle's excludes it.

Note the direction. This preregisters the *null* as the expectation, because that
is what the evidence so far predicts. A positive result would be the surprise.

## What is frozen, before generation

- **Model:** `openai/gpt-oss-120b` via Groq. A reasoning model with a different
  error distribution to Gemini-Pro-1.5, and independently sampled.
- **Problems:** 434 LiveCodeBench problems, listed in `configs/e0029_split.json`.
  434 gives ~217 evaluation problems, the count E0028's paired variance requires
  to resolve ε = 0.02.
- **Split:** `sha256(question_id)` parity, the same rule as before, applied to the
  new problem list and written to disk before the first API call.
- **Sampling:** 10 samples per problem, temperature 0.2, `max_completion_tokens`
  2500.
- **Features, target, protocol:** identical to E0028 — trajectory and static
  features, "does ANY later sample succeed given all so far failed", grouped inner
  CV on calibration, operating point frozen, applied once to evaluation.

Nothing in the pipeline is re-tuned for this model. Reusing E0028's exact
machinery is the point: a replication that also changes the method tests nothing.

## No pooling

E0029 will **not** be combined with E0028 to manufacture n = 217. They are
separate estimates on separate model distributions, and pooling them would
silently change the estimand. The comparison is descriptive:

| outcome | reading |
|---|---|
| both null | stronger evidence that observable allocation is hard in general |
| both positive | first evidence of generality |
| one positive | backend-specific, or an interaction with the task distribution |

## Resource accounting

Measured, not estimated: 1829 tokens per sample on real LiveCodeBench prompts
(a toy prompt suggested 388 — a 4.7× underestimate, recorded here because it is
the kind of error that makes a plan look cheap).

- 434 problems × 10 samples = 4340 calls ≈ **7.94M tokens**
- At the measured 8000 tokens/min limit: **≈16.5 hours** of wall clock
- Execution cost is separate and reported separately, as in E0026

Generation is checkpointed after every problem and resumable, because a run of
this length will be interrupted.

## Stop criteria

- generation cannot reach 434 problems → record what was obtained and mark the
  power shortfall rather than analysing an underpowered set as if it were whole
- any red trap
- inference over-spend > 2%
- a feature turns out to be unavailable at decision time
- the new model's solve-rate structure leaves no mixed problems, i.e. no ceiling

## What would refute the null

A learned ranker whose advantage over the matched-cost fixed baseline has a
95% CI excluding zero, with all guardrails green. That, and only that, would be
the first positive allocation result on real LLM data in this project — and even
then the claim would be tied to this benchmark, this model, this resource axis,
this split and this estimator, pending its own replication.
