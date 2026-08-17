# Preregistration — Phase 4R: structural criterion, then ceiling, then controller

**Written before the structural search runs.** The Phase 4 family is CLOSED as a
Governor benchmark: perfect-information adaptive headroom peaked at **+0.046**
(`experiments/E0004-ceiling`). That is a property of the environment, not of the
Governor, and the Env 6 result is untouched by it.

## The mandatory construction gate

**No controller is built until a family passes a ceiling gate.** This is now a
standing rule, not a step in one experiment. I had it in memory from
Environment 2 and built the whole Phase 4 stack without running it.

## Two structural quantities, both preregistered

The Phase 4 post-mortem identified two independent causes. A successor family
must fix **both**, and the second is the one an "8–12 items" rule would miss.

### S1 — competition: useful opportunities must exceed affordable upgrades

Let `X` = number of items in an episode whose realised gain from the deep budget
is positive, and `K` = number of upgrades the budget affords.

> **Require `P(X > K) >= 0.60`** and `E[X] / K >= 1.8`.

Phase 4 had `P(X > K) = 0.275` and `E[X]/K = 0.95` — supply matched demand, so
the policy was rarely forced to refuse a useful opportunity. Note that
`P(X > K) = 0.275` is already "materially above zero", which is why the
criterion is a *specified range* and not a sign test.

### S2 — decidability: reservation must not swamp preference

> **Require `mean(actual cost) / cap >= 0.70` for the DEEP mode.**

A hard budget must reserve the worst case. Phase 4's deep mode used **817 of a
2928-token reservation — 28%**, so whether an upgrade was affordable depended on
how much *earlier* calls happened to cost, not on which item deserved it.
Feasibility decided; preference did not. This is the dominant cause and it is
invisible to any criterion phrased only in items per episode.

S2 is achieved by choosing the deep mode's `max_tokens` near where the engine
actually stops, not far above it. That trades a little accuracy for a
reservation that means something.

## Search space (calibration items only, zero new API calls)

Every combination below is evaluated against **already-cached responses**:

| axis | values |
|---|---|
| items per episode | 4, 6, 8, 10 |
| LOW `max_tokens` | 300, 700 |
| HIGH `max_tokens` | 700, 1400, 2800 |
| episode budget | swept |

The configuration is chosen by S1 and S2 — **structural criteria fixed before
any ceiling was computed** — and the ceiling is then *reported*, not optimised.
Choosing the configuration with the largest ceiling would be selecting on the
outcome, which is the objection this document exists to answer.

## The ceiling gate

For the configuration selected by S1 and S2, measure through the canonical
executor and nothing else:

1. all-cheap floor
2. best fixed schedule
3. budget-limited greedy
4. fully-informed sequential oracle (exact: enumeration over subsets of
   beneficial items, which is exactly optimal because upgrading a non-beneficial
   item is weakly dominated)

> **PASS requires `U(oracle) - U(greedy) > 0.02`**, the project's existing
> frozen materiality threshold, **with a 95% CI excluding 0.02**, and the same
> result on **fresh held-out episodes** the configuration was not selected on.

The independent unit is the **ITEM**, not the episode: episodes are groupings of
a shared item pool, so a CI over episodes would be anticonservative. Intervals
come from a cluster bootstrap that resamples items and re-forms episodes.

If the ceiling fails, **reject the family immediately** and do not build a
controller. If it passes, freeze the family and the configuration, record them,
and only then build the Governor.

## What is recorded

Experiment id, commit, config, raw episodes, ceiling, budget, the
useful-opportunity distribution `P(X > K)`, the cap-to-actual ratio, the
independent-unit definition, the CI, and the decision — through
`governor/harness/ledger.py`, which refuses to finalize without them.

## Spending

**No paid quota until the ceiling passes.** The search and the gate both run on
cached responses. Groq's binding limit is TPD 200,000/day charged on *reserved*
tokens; the Phase 4 run would have needed eight days for an experiment now known
to be worthless.

## What would refute the whole approach

If **no** configuration in the search space satisfies S1 and S2 simultaneously,
then this engine and this task family cannot pose an allocation problem under a
hard reserved budget, and the correct conclusion is that the Governor needs a
different *kind* of environment — not a different controller.
