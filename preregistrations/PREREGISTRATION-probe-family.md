# Preregistration — the probe family (Environment 4)

**Status: SPECIFICATION ONLY. No code written, no runs performed.**
**Revision 2 (2026-08-16): amended under review before implementation.**
Amendments are marked [R2] and are binding exactly as the original text is.
Written 2026-08-16, after the Phase 2 stop, and deliberately fixed before
implementation so that no parameter can be chosen in response to a result.

Everything below that is marked PREREGISTERED is binding. Changing any of it
after a run invalidates that run and the change must be recorded as such.

---

## 1. The requirement, derived from measurement rather than intuition

The gated family failed for one specific, measured reason. Its clairvoyant
headroom was real (+0.153 over always-myopic), but the best observable switch
captured only 25% of it, and always-myopic *degraded* 0.725 → 0.657 → 0.644 as
regime-directed exploration was given more steps.

The mechanism: **the only evidence about the regime was block features, which
are also the only evidence about the label.** Identification was paid for in the
same currency it was trying to save. A rational controller there correctly
declines to activate itself.

So the requirement is not "add a cheap diagnostic". It is precise:

> An action that is **cheap in the budget currency**, **informative about the
> regime**, and **uninformative about the label**.

The third clause is the one the gated family violated and the one that makes the
experiment sharp.

## 2. The instrument: a probe with exactly zero task value

PREREGISTERED. Add one acquisition group, `probe`, to the existing gated family.
Acquiring it returns a single scalar

    s ~ Normal(sigma_other, sigma_probe)

and nothing else. By construction:

    I(probe ; y | c) = 0        exactly, not approximately
    I(probe ; c)     = 0        exactly
    I(probe ; sigma_other) > 0

The probe **cannot** improve label accuracy through any path. Its only possible
value is in changing a decision. This makes the central question unambiguous:

> If buying the probe is ever worth its cost, that purchase is metareasoning by
> construction, because the probe has no other use.

That property is what the gated family could not offer, and it is why this
design is worth one attempt rather than another parameter sweep.

Real-world reading: a cheap metadata inspection that characterises a task
without solving any of it — repository size and test count before deciding how
hard to look, a schema peek before deciding whether to query, a smoke check
before committing to a full run.

## 3. Preregistered parameters

| Parameter | Value(s) | Rationale, fixed in advance |
|---|---|---|
| `probe_cost` | **0.25** | [R2] A single preregistered **operating point**, fixed before evaluation. NOT justified as "cheap enough / expensive enough" — that phrasing was designer intuition dressed as reasoning. Its adequacy is not asserted; instead the analytic break-even price is computed and reported separately (§3a). No sweep. |
| `sigma_probe` | **{0.05, 0.15}** | Sharp and blunt probe. Two values only. |
| `sigma_other` train | 0.10, 0.35, 0.60, 1.50 | Unchanged from the gated family. |
| `sigma_other` test | 0.20, 0.22, 0.48, 0.90 | Unchanged. Off-grid values are misspecified by design. |
| `gate_cost` | 1.0, 2.0 | Unchanged. |
| Budget | 3, 4, 5 | Unchanged. |
| Regime prior | uniform over `REGIME_GRID` | Unchanged, preregistered, untuned. |

No parameter above may be adjusted after seeing a result. If the probe turns out
never to be worth 0.25, that is a finding, not a reason to lower it.

### 3a. [R2] The analytic break-even price, reported not tuned

The whole result could otherwise hinge on one designer-chosen number. The fix is
to derive the economics rather than rely on the number. At state s define

    C*(s) = V_probe(s) - V_no-probe(s)

the largest price at which buying the probe is justified at s. This is
computable by simulation using hidden state, so it is an environment property,
not a policy's performance.

PREREGISTERED PROTOCOL. C*(s) is computed on TRAINING configurations only and
its distribution reported as part of environment characterisation, alongside
the fraction of states with C*(s) > 0.25. `probe_cost` stays at 0.25 **whatever
that distribution shows**. Any change would be a protocol deviation and must be
recorded as one.

This also reframes the central question. It is no longer "does 0.25 work?" but:

> Does the controller buy the probe exactly on those states where
> C_probe < C*(s)?

which is a statement about the decision rule and is insensitive to the operating
point. Reported as precision/recall against the C*(s) > 0.25 indicator.

## 4. Preregistered test split

PREREGISTERED, and fixed **before** the run rather than after — the Phase 2
split was left unrepaired precisely because rebalancing after a null would have
been gate-shopping, and the honest fix is to get it right in advance.

The Phase 2 held-out set was **71% myopic-favouring** (17 of 24 configurations
had always-myopic ≥ always-strategic), which suppressed available headroom. The
new split must be **balanced by the sign of the clairvoyant advantage**: held-out
configurations shall be selected so that between 40% and 60% favour strategic,
measured by the clairvoyant arm *before* any switching policy is evaluated.

Balancing on the clairvoyant sign is legitimate because it uses no policy's
performance — only the environment's structure.

## 5. Preregistered gates — all must pass

A result counts only if **every** condition below holds. Any single failure is a
stop, reported as a negative result.

**G1 — Observability.** `P(sigma_other | nothing observed)` equals the
preregistered prior to within 1e-9, for every true regime. Same audit that the
gated family passed at 1.39e-16.

**G2 — The probe carries no label information.** [R2] Marginal independence is
not enough; the causal structure must be pinned. Three executable checks, all
required:

  G2a  A classifier given the probe value alone must not exceed the 1/8 base
       rate.
  G2b  A classifier given (probe, all block observations) must not beat one
       given (all block observations) alone. This is the one that matters: a
       probe could be marginally independent of y yet still carry label
       information once conditioned on what else was seen.
  G2c  I(probe ; c) is zero to within sampling error, so the probe cannot
       substitute for the gate.

The analytic argument in §2 must be confirmed by execution, per the environment
contract. G2b is the check that answers "the probe is secretly carrying task
information", and it is a falsification test, not a formality.

**G3 — Beats always-myopic.** `C − A > 0` with the 95% CI excluding zero, paired
across held-out configurations.

**G4 — Beats always-strategic.** `C − B > 0`, CI excluding zero.

**G5 — Beats the strongest non-cognitive baseline.** [R2] Strengthened. The
comparator is policy **E**, the best static policy-by-configuration lookup,
**fitted on TRAINING configurations only** and applied to held-out ones. Simply
beating per-configuration `max(A, B)` computed on the test set is itself gameable
by configuration memorisation, which is the failure it was meant to catch.

    U_D > U_E  on held-out configuration COMBINATIONS, CI excluding zero.

This is the decisive gate. The any-time switch cleared G3 and G4 while being a
six-entry lookup on (gate_cost, budget); E is that lookup made as strong as
possible and given a fair fit.

**G6 — Non-degeneracy.** Escalation frequency must vary *within* every
(gate_cost, budget) cell. A decision constant inside a cell is a lookup
regardless of how well it scores.

**G7 — The probe is actually bought, and selectively.** Probe purchase rate
strictly between 5% and 95%, and varying *within* every (gate_cost, budget)
cell. Always or never buying it means the decision is not state-dependent.

**G8 — [R2] The probe's value is indirect.** The scientific centrepiece, stated
as a gate so it cannot be skipped. Required jointly:

    direct task gain of the probe  = 0     (G2 establishes this)
    indirect decision gain         > 0     in some configurations

If both hold, the probe's entire contribution is to a decision, which is the
definition of metareasoning being tested.

## 5a. [R2] Policy set and the headline metric

Five policies, preregistered:

| | Policy | Sees |
|---|---|---|
| A | always cheap — never buy the strategic mechanism | — |
| B | always deliberate | — |
| C | oracle switch — knows the true value of probe and strategic action | hidden state |
| D | **Governor** — observable state only | see §5b |
| E | best static policy-by-configuration lookup, fitted on TRAIN configs | configuration key |

Target result: `U_A, U_B, U_E < U_D < U_C`. `U_D < U_C` is expected and fine.

**Headline metric: Net Value of Diagnostic Information.**

    VDI(s)     = V_with_probe(s) - V_without_probe(s)
    NetVDI(s)  = VDI(s) - C_probe

and the Governor's rule is exactly

    BUY PROBE  <=>  E[VDI | observable state] > C_probe

This is the precise economic quantity the cognitive layer is meant to learn, and
it is sharper than the generic VOI used earlier in the project because the
probe's only purpose is regime diagnosis, never task solving.

## 5b. [R2] Information the Governor may and may not receive

MAY: observed probe values, ordinary task observations, current progress,
remaining budget, action costs, its own posterior-derived uncertainty.

MAY NOT: `sigma_other`, task id, configuration key, task-family label, generator
seed, the oracle posterior, the true strategic value, or any statistic derived
from them.

The environment may use hidden state to generate training labels and oracle
values. That is a target, not an input. Enforced by test, as with the
observability contract that caught the previous leak.

## 5c. [R2] Split on unseen COMBINATIONS, not unseen values

Holding out a `sigma_other` value alone does not test whether the interaction
was learned. The held-out set must contain parameter COMBINATIONS never seen
jointly in training — e.g. train on (sigma 0.6, cost 1.0) and (sigma 1.5,
cost 2.0), test on (sigma 0.6, cost 2.0) — in addition to the off-grid sigma
values already specified in §3.

Combined with §4's balance requirement, the split is fixed before any run.

## 6. Preregistered predictions

Recorded so they can be wrong in public.

1. The probe will be bought most often at **tight budgets with expensive gates**,
   where the cost of a mistaken escalation is highest.
2. `sigma_probe = 0.05` will pass more gates than `0.15`. If the blunt probe also
   passes, the effect is robust; if neither passes, the two-currency hypothesis
   is wrong and Environment 4 is closed like its predecessor.
3. **G5 is the likely failure point**, not G3. Beating the fixed policies has
   already proven easy for degenerate reasons; beating the per-cell oracle lookup
   has not been achieved by anything yet.

## 6a. [R2] Environment 4a and 4b are separate experiments

**4a — this document.** A perfectly decoupled scalar probe. Deliberately clean:
the purpose is the cleanest possible causal test of one mechanism, so cleanliness
is a feature rather than a weakness.

**4b — not now, not mixed in.** Noisy or partially confounded diagnostics, i.e.
robustness. A pass in 4a does NOT establish robustness to realistic diagnostics,
and 4b must not be started until 4a has returned a verdict. Mixing them would
make a failure uninterpretable.

## 7. What is NOT being done

Per the standing scope agreement: no return to SynthBug or AgentCE, no further
tuning of CUBE-NM or the gated family, no POMDP solver, no RL, no LLM, no
State Manager/ActionExecutor integration. Environment 4 is one attempt against the gates above.

If it fails, the project's contribution is the four negative results and the
methodology that produced them:

1. Direct `P(success | s,a)` value modelling was the wrong abstraction.
2. Fixed-structure benchmarks collapse the switch into a budget lookup.
3. Latent strategic value can exist while being economically unrecoverable,
   when identification is paid for in the currency it saves.
4. Policies that beat both fixed baselines can still be lookup tables; the bar
   is the per-cell oracle lookup, not the fixed policies.

That is a coherent progression, and each result eliminates a class of
architecture rather than merely reporting a number.

---

## Revision 3 — PROTOCOL DEVIATION, recorded 2026-08-16

**One parameter changes. Everything else is unchanged and still binding.**

### The deviation

`Budget` becomes **{3.0, 3.5, 4.0, 4.5, 5.0}** in place of {3, 4, 5}.
`probe_cost` stays at **0.25**. No other parameter moves. No sweep is added.

### Why, and why this is not tuning

Rev2 fixed budgets at integers while block features cost 1.0. Under that
combination **any probe price in (0, 1] displaces a whole feature**:

    B=3.0:  no probe -> 3 features  |  buy probe(0.25) -> 2 features
    B=4.0:  no probe -> 4 features  |  buy probe(0.25) -> 3 features

so the probe's effective price is 1.0, not 0.25. The preregistered parameter set
cannot express "cheap" — the single concept Environment 4a exists to test. The
specification contradicted its own stated intent.

The measured consequence, committed unchanged at `fc82e82`:

    INFO value, probe free      mean +0.056, positive in 83% of configurations
    NET  value, probe at 0.25   mean -0.063, positive in 12% of configurations

The information is real and the decoupling works. The swing of ~0.12 between
those rows is the marginal value of one feature — the defect, quantified.

Three facts make this a defect repair rather than a favourable-result search:

1. **No policy result had been computed.** Only the environment was
   characterised. There was no switching number to be disappointed by.
2. **The as-specified result is committed and permanent.** It cannot be
   displaced; both versions will be reported.
3. **The change makes the environment harder, not easier.** At B=3.0 the probe
   still costs a full feature; at B=3.5 there is slack and it does not. The
   probe's *effective* cost now varies with remaining slack, so the controller
   must reason about a state-dependent price rather than a constant one.

### What would have been the alternative

Declaring 4a failed as specified. Rejected because it discards a construction
that passed every validity gate (G1, G2a, G2b, G2c) over an arithmetic oversight
in the spec rather than over a scientific finding — but it was a defensible call
and is recorded here as the road not taken.

### Unchanged and still binding

Gates G1–G8, the five-policy set, NetVDI as headline metric, the Governor
allow/deny input list, the clairvoyant-balanced split, held-out parameter
combinations, and the 4a/4b separation. `probe_cost = 0.25` remains a single
operating point with no sweep.

---

## Revision 4 — REV3 IS REFUTED. Recorded 2026-08-16.

**The rev3 budget amendment failed, and it failed on the justification I gave
for it.** Rev3 argued that half-integer budgets make the environment *harder*,
because the probe's effective cost becomes state-dependent. Measured:

    value of the buy/skip decision, accuracy units
      never buy                  +0.0000
      BUDGET-ONLY lookup         +0.0313
      oracle per-configuration   +0.0333   -> lookup captures 94%

    sign of NET by budget
      B=3.0    0% positive        B=3.5  100% positive
      B=4.0   12% positive        B=4.5   88% positive
      B=5.0   25% positive
    correlation of NET with budget parity: +0.804

The decision collapses to "buy iff the budget is half-integer". The cost did
become state-dependent — on a state variable the Governor is handed. A
one-variable lookup captures 94% of the oracle decision value.

Fifth instance of this failure mode in the project, and the first I introduced
myself while explicitly arguing it would not occur.

### The actual diagnosis

Both rev2 and rev3 fail for one underlying reason, now visible:

> In a discrete-acquisition environment with a SCALAR budget, a "cheap
> diagnostic" cannot be cheap. Its minimum effective price is one acquisition
> slot, or zero if it happens to fit in slack. There is no middle, so the probe
> is either free or costs a whole feature, and which one it is depends on budget
> arithmetic rather than on the task.

Making the price smaller cannot fix this. Two currencies require two
**resources**, not two magnitudes of one resource. That is precisely the
distinction raised in review earlier — `C_meta != C_information`, tokens versus
tool calls — and it was correct in a way neither of us followed through.

### What the corrected design must be

The probe draws from a SEPARATE resource dimension from acquisitions.
`governor/accounting/meter.py`, written in Stage 1, already supports this and it
is verified working:

    Accountant(Envelope(tool_calls=5.0, tokens=3.0))
    charge('probe',   tokens=1.0)      -> consumed tool_calls 0.0
    charge('acquire', tool_calls=1.0)

so a probe charged to `tokens` displaces no acquisition at all. The whole
CUBE-NM -> gated -> probe line has been run on a scalar budget, which is why the
two currencies kept collapsing into one. The environment work drifted away from
the project's own accounting foundation and reproduced, three times, a problem
that foundation was built to prevent.

### Status

Environment 4a is CLOSED under both rev2 and rev3 parameterisations. Its
construction gates passed (G1, G2a, G2b, G2c) and the probe's information value
is real (+0.056 free, positive in 83%), so the *instrument* is sound; the
*resource model* is not.

No further parameter amendment. Any successor must use a two-dimensional
envelope and be preregistered afresh.
