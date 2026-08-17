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
