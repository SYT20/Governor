# Preregistration — Phase 4: Governor + Nemotron

**Written before any Phase 4 API call beyond the 8-call rate probe.**
Commit this file *before* running `scripts/p4_curve.py`.

## What is being tested

Whether the allocation architecture validated on Environment 6 (a synthetic
environment with a designed latent difficulty bit) still produces value when the
deep arm is a real LLM whose competence is not a parameter I chose.

## What is different from Env 6, deliberately

Env 6 had a hidden `hard` bit and an observable cue flipped at `CUE_NOISE=0.15`.
The Governor's job was to invert a noise process I had written, at a
signal-to-noise ratio I had set. That is a fair test of the allocator and a weak
test of anything else.

Phase 4 has **no hidden difficulty variable**. The full prompt is observable, and
`n_ops` is recoverable from it by counting operators. The uncertainty the
Governor faces is not about a concealed feature — it is about **the engine's own
token-dependent competence**: whether this particular chain trips Nemotron up at
a small budget. Nobody sets that rate. It is measured.

This is a harder and more honest problem, and it is allowed to fail.

## Frozen before the curve runs

- **Engine**: `nvidia/nemotron-nano-9b-v2:free` via OpenRouter, `temperature=0`.
- **Budget lever**: total `max_tokens` (measured in Phase 3; the provider ignores
  `reasoning.max_tokens`).
- **Charged cost**: `usage.total_tokens` as reported by the provider. Never
  nominal, never `reasoning_tokens` alone, never call count.
- **Budget grid for the curve**: `[300, 700, 1400, 2800]`.
- **Curve sample**: 40 items from the calibration pool (`seed=1000`).
- **PROMPT_CAP = 128** tokens. Asserted against measured `prompt_tokens`; if any
  prompt exceeds it the run aborts rather than silently overspending.

### Mode selection rule (frozen)

Let `acc(b)` be mean correctness on the curve sample at budget `b`.

1. `HIGH` := the smallest grid `b` with `acc(b) >= max_b acc(b) - 0.02`.
2. `LOW`  := the largest grid `b < HIGH` with `acc(b) <= acc(HIGH) - 0.15`.
3. Contingency, specified in advance: if the smallest grid point is already
   within 0.15 of `acc(HIGH)`, extend the grid downward by halving, at most
   twice, and record that this happened.
4. If no `LOW` exists after the contingency, **the Phase 4 premise fails at this
   task family** — there is no budget region where allocation can matter. Report
   that and stop. Do not tune the task family to create a gap.

### Episode budget rule (frozen)

```
cap(m)       = PROMPT_CAP + m                 # worst-case charge for one call
TOTAL_BUDGET = 4*cap(LOW) + 2*(cap(HIGH) - cap(LOW))
```

Four items, enough slack for exactly two upgrades in the worst case. This mirrors
Env 6's "2 deep calls out of 4" without importing any of its numbers.

Feasibility is checked against `cap(m)`; the charge is the actual
`total_tokens`. A policy therefore cannot overspend, and unspent budget is
reported as utilization rather than quietly reallocated.

---

## AMENDMENT 1 — 2026-08-17, after the E0001 curve, before any policy comparison

**Status of the split at the time of writing: the test pool has not been
generated, no policy has been executed, and no held-out number exists.** Only
the 40-item calibration curve has been seen. This amendment changes the budget
formula and nothing else. The engine selection, mode selection, primary
hypothesis, and pass criterion are unchanged.

### What the curve showed

| budget | acc | starved | mean total_tokens | cap | used / cap |
|---|---|---|---|---|---|
| 300 | 0.050 | 95% | 369 | 428 | 86% |
| 700 | 0.525 | 48% | 665 | 828 | 80% |
| 1400 | 0.950 | 5% | 784 | 1528 | 51% |
| 2800 | 1.000 | 0% | 817 | 2928 | 28% |

The frozen rule selects LOW=700, HIGH=2800, and the frozen budget formula gives
`4*828 + 2*(2928-828) = 7512`.

### The defect

The formula reserves each call at its **cap**, and assumed a call costs roughly
its cap. At HIGH the model stops after 817 tokens of a 2928-token reservation —
28%. Under-spend returns the difference to the pool, so:

```
B = 5000  greedy realises 1 deep call of 4
B = 5200  greedy realises 2
B = 5300  greedy realises 3
B = 5412  greedy realises 4
B = 7512  greedy realises 4      <- the preregistered budget
```

At 7512 **the budget does not bind**. Greedy, all-deep, and the Governor all
upgrade every item, the oracle is matched by a fixed schedule, and the
experiment measures nothing. This is a defect in the formula, not a result.

### The amendment

The formula's stated intent was "room for exactly two upgrades out of four". It
expressed that in worst-case reservations, which are not what a call costs.
Restate it in realised terms:

> **TOTAL_BUDGET** := the smallest `B` on the grid
> `range(4*cap_low, 4*cap_low + 4*(cap_high - cap_low), 50)` such that the
> greedy policy's **mean realised deep calls on the CALIBRATION split** is
> `>= 2.0`.

Computed on calibration only. Feasibility still reserves the cap, so the hard
budget remains provably un-violable; only the total changes.

### Why this is not tuning

The quantity being set is the **scarcity of the resource**, which is the
experiment's independent variable, not its outcome. It is chosen by a stated
rule on the calibration split before the test pool is touched, and the criterion
("greedy gets about half the items") is the same criterion the original formula
was trying to express. Nothing about which policy wins enters the choice.

The budget also remains a **swept dimension** in the Step 7 robustness sweep
(0.6x - 1.6x), so the result is reported across scarcity levels rather than at
one chosen point.

### Engine

Nemotron returned `X-RateLimit-Limit: 50, Remaining: 0`,
`limit_source: openrouter_free_tier_daily` — 50 free requests per day against a
requirement of ~1400. It produced no curve, so it does not qualify under the
frozen rule and the selection falls through to **qwen/qwen3.6-27b**. No
amendment is needed for this; the rule already excludes throttled engines.

---

## Splits (frozen)

| split | pool seed | purpose |
|---|---|---|
| calibration | 1000 | curve, mode selection, best fixed schedule, value predictor |
| test | 20260817 | held-out, touched once |

Pools are generated from disjoint seeds and item ids are seed-prefixed, so no
test item can appear in calibration. Episodes group four pool items; the
grouping seed is separate from the pool seed.

## Primary hypothesis and pass criterion (frozen)

**H1**: `U(Governor + Nemotron) > U(best fixed Nemotron policy)` on the held-out
split, at an identical total-token budget.

PASS requires **all** of:

1. mean paired difference > 0 with a 95% CI excluding zero;
2. identical `TOTAL_BUDGET` for every policy, enforced by the canonical executor;
3. no red trap check;
4. the experiment finalizes through `record_experiment` (raw rows present,
   commit clean, config hash matching).

Secondary, reported whether or not H1 passes: Governor vs greedy, vs the text
heuristic, vs all-cheap, and the fraction of oracle headroom captured.

## Common random numbers

Every `(item, budget)` pair is called **once** and cached; all policies read the
same recorded response. This is the same device as Env 6's shared `roll` array:
it removes sampling noise from the comparison so the difference between policies
is the decision and not the draw. It is applied identically to every policy,
including the oracle.

## What would refute the architecture

- The best fixed schedule matches the Governor (allocation carries no value on
  real items).
- The Governor's decisions are constant across observable states
  (`constant_schedule` trap goes red).
- Correctness at `LOW` and `HIGH` differ by less than 0.15 anywhere on the grid
  (no allocation problem exists).

Any of these is reported as a Phase 4 failure. The task family will not be
regenerated to avoid one.
