# Preregistration — Environment 5: Two-Currency Metareasoning

**Status: SPECIFICATION ONLY. No code written, no runs performed.**
Written 2026-08-16 after Environment 4a was closed under both parameterisations.

Everything marked PREREGISTERED is binding. Changes after a run invalidate that
run and must be recorded as deviations, as rev3/rev4 of the 4a document were.

---

## 1. What 4a established, and what it did not

**Sound and carried forward:** the probe instrument. It passed G1 (observability
exact), G2a, G2b (no label information conditional on block observations) and
G2c. Its information value is real — **+0.056 mean, positive in 83% of
configurations** when granted free.

**Refuted:** the scalar resource model, twice.

    integer budgets      probe priced 0.25 has effective price 1.0 (one slot);
                         info +0.056 becomes net -0.063, positive in 12%
    half-integer budgets decision collapses to "buy iff budget is half-integer";
                         a budget-only lookup captures 94% of oracle decision
                         value; corr(NET, parity) = +0.804

**The finding:** in a discrete-acquisition environment with a scalar budget, a
cheap diagnostic cannot be cheap. Its minimum effective price is one action slot
or zero. Lowering the nominal price cannot fix quantisation. Two currencies
require two **resources**, not two magnitudes of one.

## 2. [CORRECTED] No conversion between accuracy and resources

The 4a preregistration made `NetVDI = VDI - C_probe` its headline metric. That
is **dimensionally meaningless**: VDI is in accuracy points and `C_probe` is in
budget units. It is the same units error that produced a wrong conclusion twice
in this project, written into the document intended to prevent it. Retracted.

PREREGISTERED — **Option A, hard resource constraints. No shadow prices.**

    maximise  E[task success]
    subject to  tokens     <= B_think
                tool_calls <= B_tool

Utility stays in one unit (task success). Resources are constraints, never
terms in the objective. No λ needs justifying, and no accuracy-per-token
exchange rate is invented.

Shadow prices (`U = success - λ_T·tokens - λ_C·calls`) are explicitly rejected:
every λ is another unjustified parameter, and this project's failures have come
from exactly such numbers.

## 3. The resource split

PREREGISTERED. Uses `governor/accounting/meter.py` from Stage 1 — the existing
`Envelope(tokens, cost, wall_s, tool_calls)` and `Accountant.charge(label,
**amounts)`. Verified working: charging `tokens=1.0` consumes `tool_calls=0.0`.

No bespoke budget abstraction. The environment work drifted away from this layer
and reproduced, three times, a problem it was built to prevent.

| Action | Charges | Rationale |
|---|---|---|
| acquire a task feature | `tool_calls += 1` | external information |
| probe (regime diagnostic) | `tokens += 1` | internal deliberation |

The probe consumes **no** `tool_calls`. That is the whole point and it is
structural rather than a matter of price.

## 4. THE CRITICAL DESIGN QUESTION — flagged, not decided

**A separate reasoning budget with no alternative use makes the experiment
degenerate.** If `B_think` can only be spent on the probe and unspent think
budget is worthless, then spending it is free, always-deliberate wins by
construction, and we have rebuilt the "always escalate" degeneracy in a new
costume. Environment 4a died of an unexamined resource assumption; this is the
same hazard at the next level and it must be closed **before** implementation.

Three candidate closures. **I am not choosing among them unilaterally** — the
last two amendments I made alone both failed, and the second failed on the
reasoning I gave for it.

**(a) Competing deliberations.** Several probes exist, each informative about a
different aspect; `B_think` affords only some. Spending on one forgoes another.
Makes think allocation itself a decision. Risk: turns into a second acquisition
problem rather than a metareasoning one.

**(b) Shared budget across a task sequence.** One `B_think` spans N tasks.
Spending early means less later. Closest to the real deployment setting and to
the original "budget-controlled cognitive layer" framing. Risk: adds a credit
assignment problem across tasks.

**(c) Coupling through a third resource.** Both actions also consume `wall_s`,
under a latency constraint, at different rates. `Envelope` already carries
`wall_s`. Risk: reintroduces a scalar bottleneck through the back door, which is
precisely what killed 4a — this one should probably be rejected for that reason.

**Recommendation: (b).** It is the only one where the think budget's opportunity
cost is genuine and external to the current decision, and it matches what the
project set out to build. But it is a design choice with the power to
manufacture a result, so it is written here for review rather than adopted.

## 5. PREREGISTERED policy set

| | Policy | Sees |
|---|---|---|
| A | never deliberate | — |
| B | always deliberate | — |
| C | oracle metareasoner | hidden state |
| D | **Governor** | observable state + remaining resource **vector** |
| E | strongest non-cognitive baseline: static policy per configuration, fitted on TRAINING configurations only | configuration key |

**PRIMARY criterion: `U_D > U_E`** on unseen configuration combinations, CI
excluding zero. Not `U_D > U_A`.

This project has produced two policies that beat both fixed baselines while
being lookup tables — the any-time switch captured 94% of its value from a
six-entry (cost, budget) table. E is that table made as strong as possible and
given a fair fit, so beating it is the claim that means something.

## 6. PREREGISTERED gates

**H1** Observability: regime posterior before any observation equals the
preregistered prior to 1e-9, every regime. (4a passed at 0.00e+00.)

**H2** Probe carries no label information: H2a probe alone at chance; H2b
probe+observations no better than observations alone; H2c I(probe; context)
within a permutation null. Carried over unchanged — 4a passed all three.

**H3** `U_D > U_E`, CI excluding zero. **Primary.**

**H4** `U_D > U_A` and `U_D > U_B`. Necessary, not sufficient.

**H5** Non-degeneracy: probe purchase rate strictly inside (5%, 95%) **and
varying within every cell of the coarse resource variables**. A decision
constant inside a cell is a lookup regardless of its score.

**H6** Resource separation holds in execution: probing must be verified to leave
`tool_calls` unchanged. Asserted by test, not by comment — 4a's observability
leak entered exactly through a claim made in prose about code that did
otherwise.

**H7** No scalar collapse: the buy/skip decision must not be predictable from
any single resource dimension. Reported as the fraction of oracle decision value
captured by the best one-variable lookup; **must be below 70%**. The 4a rev3
failure scored 94% here and this gate would have caught it.

## 7. Explicitly out of scope

No further 4a variants. No probe-cost tuning of any kind. No shadow prices. No
LLM, Graft, or Ares integration yet — though the token dimension is chosen so
that `tokens` later corresponds to real LLM inference cost rather than a
synthetic number, and `tool_calls` to real tool invocations.

Environment 5 is one attempt against H1–H7. If it fails, the project's
contribution is six negative results and the methodology that produced them.
