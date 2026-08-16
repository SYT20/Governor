# Preregistration — the probe family (Environment 4)

**Status: SPECIFICATION ONLY. No code written, no runs performed.**
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
| `probe_cost` | **0.25** | One quarter of a block feature. Cheap enough to be plausibly worth it, expensive enough that buying it always is not free. Single value — no sweep, so it cannot be tuned. |
| `sigma_probe` | **{0.05, 0.15}** | Sharp and blunt probe. Two values only. |
| `sigma_other` train | 0.10, 0.35, 0.60, 1.50 | Unchanged from the gated family. |
| `sigma_other` test | 0.20, 0.22, 0.48, 0.90 | Unchanged. Off-grid values are misspecified by design. |
| `gate_cost` | 1.0, 2.0 | Unchanged. |
| Budget | 3, 4, 5 | Unchanged. |
| Regime prior | uniform over `REGIME_GRID` | Unchanged, preregistered, untuned. |

No parameter above may be adjusted after seeing a result. If the probe turns out
never to be worth 0.25, that is a finding, not a reason to lower it.

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

**G2 — The probe carries no label information.** Empirically: a classifier given
the probe value and nothing else must not exceed the 1/8 base rate. This is a
falsification test of the construction, not a formality — the analytic argument
in §2 must be confirmed by execution, per the environment contract.

**G3 — Beats always-myopic.** `C − A > 0` with the 95% CI excluding zero, paired
across held-out configurations.

**G4 — Beats always-strategic.** `C − B > 0`, CI excluding zero.

**G5 — Beats the budget lookup.** `C >` per-configuration `max(A, B)`, CI
excluding zero. **This is a new and higher bar than Phase 2 used.** The any-time
switch cleared G3 and G4 while being a six-entry lookup on (gate_cost, budget);
G5 is what that policy failed, and it is the condition that actually
distinguishes a cognitive layer from a table.

**G6 — Non-degeneracy.** Escalation frequency must vary *within* every
(gate_cost, budget) cell. A decision constant inside a cell is a lookup
regardless of how well it scores.

**G7 — The probe is actually bought, and selectively.** Probe purchase rate
strictly between 5% and 95%, and varying with observed state. Always or never
buying it means the decision is not state-dependent.

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

## 7. What is NOT being done

Per the standing scope agreement: no return to SynthBug or AgentCE, no further
tuning of CUBE-NM or the gated family, no POMDP solver, no RL, no LLM, no
Graft/Ares integration. Environment 4 is one attempt against the gates above.

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
