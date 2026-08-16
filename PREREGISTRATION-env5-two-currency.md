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

## 4. [REV2] The reasoning ladder — §4's degeneracy, closed

Rev1 flagged that a think budget with no alternative use is degenerate, and
recommended sharing it across a task sequence. That is superseded. A task
sequence adds inter-task credit assignment, and a reviewer could then fairly say
"you built a budget scheduler, not a cognitive controller" — the two are related
but not the same, and the ambiguity would be unresolvable after the fact.

PREREGISTERED instead: **one shared think budget, three reasoning modes, chosen
at every decision point within a single task.**

    M0  execute            0 tokens   act on the current posterior, myopically
    M1  light deliberation c1 tokens  the regime probe from Env 4a
    M2  deep deliberation  c2 tokens  multi-step lookahead over acquisitions

with `c1 < c2`, all three drawing on the same `B_think`.

### 4a. [REV2] The ladder is degenerate with ONE decision point — stated, because
it is the trap the previous two environments fell into

If a task offers a single opportunity to deliberate, the optimal policy is
"choose the deepest mode you can afford", which is a function of `B_think`
alone: a lookup, and exactly the failure that closed Env 4a rev3.

The ladder is non-degenerate **only because `B_think` is shared across the
MULTIPLE acquisition steps within one task.** Spending M2 at step 1 means it
cannot be afforded at step 4, so the opportunity cost is internal to the task
and genuine without any inter-task machinery. This is load-bearing and must be
enforced in the implementation, not assumed:

> PREREGISTERED: every task has at least 3 decision points, and `B_think` is
> strictly less than the cost of invoking M2 at every one of them.

Without that inequality the budget does not bind and the experiment is void.

### 4b. [REV2] Token costs are DERIVED, not invented

Review required that `c1` and `c2` not be arbitrary numbers. They are not; they
are counted from the computation each mode actually performs, in units of
posterior evaluations over the H hypotheses:

    M0  act on the current posterior                        0 evaluations
    M1  probe: one 1-D quadrature over the regime marginal  ~T evaluations
    M2  k-step lookahead: candidates^k posterior rollouts   ~G^k evaluations

so `c1` and `c2` are read off the instrumented implementation rather than
chosen. The ratio `c2/c1` is then a property of the algorithms, and when an LLM
is later substituted, `tokens` maps onto real inference tokens with no change of
meaning. Any hand-set token cost would reintroduce precisely the kind of
unjustified parameter that produced this project's failures.

## 5. [REV2] PREREGISTERED policy set

| | Policy | Sees |
|---|---|---|
| A | never deliberate — always M0 | — |
| B | always deepest affordable mode | — |
| C | oracle metareasoner | hidden state |
| D | **Governor** | observable state + remaining resource vector |
| E1 | static policy per configuration, fitted on TRAIN configs | configuration key |
| E2 | **resource-only lookup** — mode as a function of (B_tool, B_think) | resources |
| E3 | **resource + progress lookup** | resources, step count |

E2 and E3 added under review. A non-scalar resource *vector* can itself support
structured lookup behaviour, so "not a scalar lookup" is no longer sufficient.

**PRIMARY criterion: `U_D > max(U_E1, U_E2, U_E3)`** on unseen configuration
combinations, CI excluding zero.

## 6. [REV2] PREREGISTERED gates

**H1 Observability.** Regime posterior before any observation equals the
preregistered prior to 1e-9. (Env 4a passed at 0.00e+00.)

**H2 Resource separation, verified in execution.** Invoking M1 or M2 must leave
`tool_calls` unchanged; acquiring must leave `tokens` unchanged. Asserted by
test, never by comment — Env 4a's leak entered through prose describing code
that did otherwise.

**H3 The reasoning modes do not solve the task directly.** M1 carries no label
information (Env 4a's G2a/G2b/G2c, all passed, carried over unchanged). M2 may
improve the *choice* of acquisitions but must acquire nothing itself.

**H4 Every mode is optimal somewhere.** There must exist held-out states where
M0 beats M1, states where M1 beats M0, and states where M2 beats M1. If any mode
is never optimal, the ladder has fewer real rungs than it claims.

**H5 No single-variable lookup explains the decisions.** The best one-variable
lookup — over any resource dimension or progress variable — must capture **less
than 70%** of oracle decision value. Env 4a rev3 scored 94% and passed every
gate that existed at the time.

**H6 Beats the strongest structured baseline.** `U_D > max(U_E1, U_E2, U_E3)`,
CI excluding zero, on unseen configuration combinations.

**H7 [REV2] Action distribution varies WITHIN identical resource states.** For
fixed `(B_tool, B_think)`, the mode chosen must vary across observable task
states on held-out configurations. A policy constant inside a resource cell is a
lookup however well it scores — this is the direct analogue of the
"constant within cost/budget cell" failure that exposed the any-time switch.

**H8 [REV2] Cheaper than always-deep.** `tokens(D) < tokens(B)` while
`U_D >= U_B`. Without this, "Governor = always deepest mode" passes H6 by
spending more, which is the opposite of a budget-controlled layer.

## 6a. [REV2] First construction gate, before any policy work

Per review, the first thing built and run is not a policy but a check on the
environment:

> Across held-out task states there must exist genuine states where each of M0,
> M1 and M2 is optimal, and no resource-only or progress-only lookup may capture
> most of the decision value.

If that fails, Environment 5 is closed before any Governor exists — the same
sequencing that made the Env 4a closure cheap and unambiguous.

## 7. Explicitly out of scope

No further 4a variants. No probe-cost tuning of any kind. No shadow prices. No
LLM, Graft, or Ares integration yet — though the token dimension is chosen so
that `tokens` later corresponds to real LLM inference cost rather than a
synthetic number, and `tool_calls` to real tool invocations.

Environment 5 is one attempt against H1–H7. If it fails, the project's
contribution is six negative results and the methodology that produced them.
