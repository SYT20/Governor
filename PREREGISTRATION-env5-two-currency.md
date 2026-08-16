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

## 3. [REV3] The resource split — compute units, NOT tokens

Rev2 claimed the derived costs "map onto real inference tokens with no change of
meaning". **That is false and is retracted.** A posterior evaluation is an
algorithmic operation; an LLM token is an inference-resource unit involving
orders of magnitude more computation, at a ratio that depends on model
architecture. Relabelling one as the other is the same units error that has now
produced a wrong conclusion three times in this project — twice in analysis, and
here in a specification written to prevent exactly that.

PREREGISTERED resource vector:

    R = (B_tool, B_compute)

    B_tool     acquisitions, in tool calls          -> Envelope.tool_calls
    B_compute  deliberation, in COMPUTE UNITS       -> Envelope.cost

`Envelope.cost` is used rather than `Envelope.tokens` deliberately, so that
`tokens` stays free to mean actual LLM tokens when an executor is attached.
The honest mapping is a research question, not an identity:

    Environment 5   compute units (instrumented primitive operations)
    Real agent      LLM inference tokens
    Relationship    to be MEASURED when the two are run side by side

Stating it that way is also a better result than pretending they are the same:
the architecture is resource-agnostic, and which physical resource fills the
second slot becomes an empirical question rather than an assumption.

## 4. [REV3] Three computational modes, one shared compute budget

Not a "ladder" — rev2's word was misleading. M1 and M2 are **qualitatively
different computations**, not two depths of one algorithm:

    M0  direct action            0 units   act on the current posterior
    M1  diagnostic deliberation  c1 units  "what regime am I in?"  (the probe)
    M2  strategic planning       c2 units  "what acquisition sequence?" (lookahead)

M1 asks a question about the task; M2 asks a question about the plan. Both draw
on the same `B_compute`, which is what makes the allocation a decision.

**M2 may not acquire anything.** It changes which acquisition is chosen and
charges `B_compute`; the acquisition itself still charges `B_tool`. This keeps
reasoning and execution structurally separate, which is what later maps onto
Governor → planner → executor rather than letting the planner quietly do the task.

### 4a. Non-degeneracy conditions, binding

With a single decision point the optimal policy is "deepest affordable mode", a
function of `B_compute` alone — the lookup that closed Env 4a rev3 at 94%.

> PREREGISTERED: at least 3 decision points per task, and
> `c2 <= B_compute < 3 * c2`, so M2 is affordable at least once and never at
> every step. Verified by assertion at construction time, not assumed.

### 4b. [REV3] Cost accounting is frozen before any policy runs

Rev2 said costs are "counted from the computation performed", which review
correctly identified as still a hand-designed convention unless the counted set
is fixed in advance. PREREGISTERED primitive counters, and no others:

    likelihood_evals     calls to loglik_cols
    posterior_updates    normalisations of the H-vector
    candidate_evals      per-candidate expected-entropy evaluations
    branch_expansions    lookahead nodes expanded

`C(M0)`, `C(M1)`, `C(M2)` are reported as exact 4-vectors from an instrumented
run and **frozen before any policy is evaluated**.

### 4c. [REV4] The unweighted sum is retracted

Rev3 collapsed the 4-vector by an unweighted sum. Review is right that this
smuggles in `1 likelihood_eval = 1 posterior_update = 1 candidate_eval =
1 branch_expansion`, which is certainly false and is a cost model wearing the
clothes of a measurement. Retracted.

PREREGISTERED SELECTION RULE — the rule is fixed here, the answer is not:

> `B_compute` is measured in the single primitive that accounts for the largest
> share of wall-clock time in an instrumented profile of M1 and M2, measured
> **before any policy evaluation**. The other three counters are retained as
> telemetry and reported, but do not constrain.

Profiling is characterisation, not tuning: it observes runtime, not results.

**[REV4] COVERAGE CONSTRAINT, which the dominance rule alone does not give.**
A primitive can dominate runtime while being consumed by only one mode — if
branch expansions dominate and M1 performs none, then M1 costs zero, the budget
does not constrain it, and "always diagnose" becomes free. That is a new
degeneracy of exactly the kind this project keeps producing.

> PREREGISTERED: the selected primitive must be consumed in non-zero amounts by
> BOTH M1 and M2. M0 must cost zero by design. If no primitive satisfies
> dominance and coverage together, the vector formulation
> (`L <= B_L, U <= B_U, E <= B_E, X <= B_X`) is used instead, and that fallback
> is chosen by this rule rather than by preference.

### 4d. [REV4] M2 search depth is frozen

`M2` uses a fixed lookahead depth `k = 2`, preregistered. If Governor could
choose `k`, M2 would be a family of modes and Environment 5 would be asking two
questions at once — "should I deliberate?" and "how deeply?" — which cannot be
disentangled after the fact. Variable depth is Environment 6, if Environment 5
passes.

## 5. [REV3] Policy set

| | Policy | Sees | Role |
|---|---|---|---|
| A | always M0 | — | floor |
| B | always deepest affordable | — | floor |
| C | clairvoyant oracle | hidden state | ceiling |
| D | **Governor** | observable state + resource vector | the claim |
| E2 | resource-only lookup | `(B_tool, B_compute)` | baseline |
| E3 | resource + progress lookup | + step count | baseline |
| E4 | resource + progress + preregistered cheap state summaries | + posterior entropy, margin | baseline |

**E1 is REMOVED from the held-out comparison.** Review is
right that it is undefined there: on an unseen combination the key does not
exist, so E1 is either undefined or is secretly a nearest-neighbour model, and
those are different baselines. E1 is reported separately as an **in-distribution
ceiling**, never as a held-out comparator.

**PRIMARY: `U_D - max(U_E2, U_E3, U_E4) > 0`** on unseen configuration
combinations, CI excluding zero.

E4 is the hard one: it gets resources, progress, and cheap state summaries. If D
cannot beat E4, then whatever D has learned is expressible as a shallow function
of quantities already available for free, and the cognitive layer is not earning
its keep.

## 6. [REV3] Gates

**H1 Observability.** Regime posterior before observation equals the
preregistered prior to 1e-9. (Env 4a: 0.00e+00.)

**H2 Resource separation, verified in execution.** M1/M2 leave `tool_calls`
unchanged; acquisitions leave `cost` unchanged. By test, never by comment.

**H3 Modes do not solve the task.** M1 carries no label information (Env 4a
G2a/b/c, carried over). M2 acquires nothing.

**H4 [REV4] Every mode is optimal often enough, with prevalence AND bounds.**
Rev3 required a configuration family where each advantage's CI excludes zero.
Review is right that one anomalous family could satisfy that. PREREGISTERED,
fixed before running:

> Each of `M0 > M1`, `M1 > M0`, `M2 > M1` must hold with a cluster-bootstrap CI
> excluding zero in **at least 10% of held-out configuration families**.

A mode that wins in one narrow corner is not a mode, it is an artefact.

**H5 [REV3] No single-variable lookup, with a FROZEN fitting protocol.** "Best
one-variable lookup" is meaningless without specifying the family, and would
otherwise become a hidden tuning exercise. PREREGISTERED: for each candidate
scalar x in {B_tool, B_compute, step, posterior entropy, margin}, fit a
one-dimensional threshold policy with **at most 4 bins at fixed quantile
boundaries {0.2, 0.4, 0.6, 0.8}**, on TRAINING configurations only, evaluated on
held-out. Fraction of oracle decision value captured by the best such policy
must be **< 70%**, reported with a cluster bootstrap over configurations.
Env 4a rev3 scored 94% and passed every gate that existed at the time.

**H6 Beats the strongest structured baseline.** The primary criterion above.

**H7 [REV3] Variation must be informative, not merely present.** Rev2 required
the mode choice to vary within identical resource cells — but random variation
passes that. Required instead: `I(action ; state | resources) > 0` on held-out
data against a permutation null, **and** the variation must improve held-out
utility. H7 remains a degeneracy detector; H6 is the actual claim.

**H8 Cheaper than always-deep.** `compute(D) < compute(B)` while `U_D >= U_B`.
Without it, "Governor = always deepest" passes H6 by spending more, inverting
the point of a budget-controlled layer.

## 6a. Construction gate first, before any Governor exists

The first thing built and run is H1–H5 on the environment alone. If there is no
held-out state where each mode is defensibly optimal, or if a resource-only or
resource+progress policy explains most of the decision value, Environment 5
closes before a Governor is written — the sequencing that made the Env 4a
closure cheap rather than a sunk cost.

## 7. Explicitly out of scope

No further 4a variants. No probe-cost tuning of any kind. No shadow prices. No
LLM, Graft, or Ares integration yet — though the token dimension is chosen so
that `tokens` later corresponds to real LLM inference cost rather than a
synthetic number, and `tool_calls` to real tool invocations.

Environment 5 is one attempt against H1–H7. If it fails, the project's
contribution is six negative results and the methodology that produced them.
