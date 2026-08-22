# E0030 — Difficulty provenance audit

**Verdict: `CONTEST_RATING_DERIVED` — admissible at decision time.**

No Governor was trained. No evaluation problem was scored. No threshold was
tuned. The 250/225 split is unchanged.

## Why the audit was needed

The corrected E0029 analysis found `difficulty` scoring **AUC 0.707** at the
allocation point, against `code_lines` at 0.651 — and `difficulty` costs nothing
to observe, while `code_lines` costs a full generation. That makes it the
obvious feature to preregister, and exactly the kind of thing that should not be
adopted because it looks good.

`difficulty` also sits on the project's `FORBIDDEN_FEATURES` list. That list was
written for the synthetic Phase-5 family, where difficulty was **the hidden
latent the Governor was supposed to infer**. Whether the same word means the
same thing on LiveCodeBench is a question about provenance, not about the word.

## The question that decides it

Not "is this metadata" but **when was the value fixed, relative to the outcome
being predicted?**

    fixed BEFORE any model touched the problem   ->  prior information, usable
    computed FROM the outcomes being predicted   ->  the label in disguise

This distinction is not hypothetical here. LiveCodeBench's published `metadata`
field turned out to *be* the label on inspection: 1530 empty/PASS, 2470
non-empty/FAIL, **zero off-diagonal**.

## What was established

### 1. The paper states the mechanism (arXiv:2403.07974)

* **LeetCode** — *"The platform also provides a difficulty label for each problem
  which we use to tag the problems as Easy, Medium, and Hard."*
* **AtCoder** — *"The problems are assigned numeric difficulty ratings, and we
  exclude abc problems with a rating of more than 500."* Brackets
  `[0-200), [200-400), [400-500]`.
* **CodeForces** — *"...using the rating brackets {800}, (800−1000], and
  (1000−1300] respectively."*

### 2. The runner reads the field, it never computes it

`lcb_runner/benchmarks/code_generation.py` defines `Difficulty` as a plain enum
and `__post_init__` does only `Difficulty(self.difficulty)`. The scraping and
dataset-construction code is **not** in the public repository — only
`lcb_runner`. So the documentation could not be verified against construction
source, which is why the next step matters.

### 3. Empirical test, on the data itself

If AtCoder difficulty came from kenkoooo/AtCoderProblems — whose difficulty is
*"the rating that 50% of people with that rating can solve the problem"*, i.e.
**human solve-rate derived** — the mapping to problem position would be noisy,
and the paper's cutoff of 500 would exclude nearly the whole scale, which runs to
~4000.

Difficulty against AtCoder problem letter, 267 problems:

| letter | easy | medium | hard |
|---|---|---|---|
| a | **51** | 0 | 0 |
| b | 48 | 5 | 0 |
| c | 0 | **52** | 0 |
| d | 0 | 34 | 17 |
| e | 0 | 0 | **47** |
| f | 0 | 0 | **13** |

Monotone in letter. Only `b` and `d` straddle a boundary — precisely what
**setter-assigned point values** produce, since ABC point values shift slightly
between contests. This is the signature of contest metadata, not of solve-rate
estimation.

### 4. The schema contains nothing outcome-shaped

    contest_date, contest_id, difficulty, metadata, platform,
    question_id, question_title

No solve rate. No pass rate. No submission count. No model result. `metadata`
holds only `func_name`, for LeetCode's functional harness.

## Answers

| # | Question | Answer |
|---|---|---|
| 1 | Who assigned it | Platforms and problem setters; LCB re-buckets, does not estimate |
| 2 | When | At or before contest publication |
| 3 | Inputs | LeetCode native label; AtCoder point values; CodeForces ratings |
| 4 | Depends on model performance | **No** |
| 5 | Depends on hidden-test outcomes | **No** |
| 6 | Depends on post-hoc benchmark eval | **No** |
| 7 | Available before the first sample | **Yes** |
| 8 | Would an agent have it | Yes for benchmark eval; not guaranteed in deployment |
| 9 | Stable across releases | **Unverified** — consistent within the cached snapshot only |
| 10 | Official field | Yes |

## Caveats, recorded rather than smoothed over

**CodeForces ratings are computed from human contestant performance after the
contest.** Human, not model, and independent of LiveCodeBench's hidden tests —
but it is outcome-derived at one remove, from a different population. Affects
**9 of 475** problems (1.9%).

**Cross-release stability is unverified.** Any experiment must pin the dataset
snapshot; the grading stage already records a sha256 per shard.

**Deployment availability.** A Governor that reads `difficulty` needs that
metadata at decision time. True for benchmark evaluation, not guaranteed for a
novel problem in production. Any generalisation claim is contingent on it.

## Consequence

`difficulty` is admissible, so a new preregistered experiment may be designed
around it. It is **not** thereby validated as a good feature — the audit
establishes only that using it is not cheating.

The name-based `FORBIDDEN_FEATURES` list should not simply be edited: it is
correct for the synthetic family, where difficulty *is* the hidden latent.
Admissibility is per-dataset, which is why it now lives in a manifest that
records a source and a time, rather than in a list of words.

## Guardrail

`governor/harness/provenance.py` declares every decision-time feature with its
source and the moment its value was fixed. `tests/test_feature_provenance.py::
test_feature_provenance_manifest` fails if any feature in the vector is
undeclared or inadmissible. **UNKNOWN is a rejection, not a default** — adding a
feature without auditing it now breaks the build rather than passing silently.
