# Governor — Consolidated Findings

A reasoning-aware, budget-controlled cognitive policy layer for LLM agents.
Status as of 2026-08-16. Every number below is reproducible from a script in
`scripts/` and a JSON in `results/`; commit hashes are given for the load-bearing
claims.

---

## 1. What is verified

**Non-myopic acquisition beats exactly-computed Bayes-optimal myopic acquisition
under tight budgets.** Measured on a reference-checked reproduction of AFABench's
CUBE-NM, 5 seeds × 400 rows, all arms sharing one exact Bayes predictor:

| budget | myopic (exact) | non-myopic | Δ | 95% CI |
|---|---|---|---|---|
| 1 | 0.159 | 0.118 | −0.041 | [−0.064, −0.018] |
| 3 | 0.328 | 0.594 | **+0.267** | [+0.239, +0.295] |
| 4 | 0.549 | 0.750 | +0.201 | [+0.196, +0.206] |
| 8 | 0.957 | 0.968 | +0.010 | [+0.006, +0.014] |

Information ceiling 0.970. The advantage is an **inverted U** — negative at
budget 1, peaking at budget 3, closing by budget 8. It is a tight-budget
phenomenon, not a growing one.

The myopic baseline here is not a heuristic. It is the Bayes-optimal one-step
policy under the true generative posterior, computed **exactly** — 1-D
quadrature per candidate, a 5-term collapse for the near-deterministic context
observation, no Monte Carlo. Audited across 150 states in 6 posterior regimes:
**150/150 argmax agreement**, 1 exceedance in 7,100 comparisons at 4 SE against
~0.4 expected by chance (`c06875f`).

---

## 2. Seven negative results

Each eliminates a class of architecture rather than reporting a number.

**N1 — `P(success | s, a)` is the wrong abstraction.** Pooled AUC 0.84 was
dominated by state-to-state variation; within-state action ranking was near
chance. Predicting *advantage* rather than outcome fixed the estimand.

**N2 — Fixed-structure benchmarks collapse the switch into a budget lookup.** In
CUBE-NM, state-feature variance at t=0 is **3.01e-27** — every instance presents
an identical posterior before the first acquisition, so no step-0 decision can
condition on anything but budget.

**N3 — A cognitive residual can be a progress variable in disguise.** An apparent
+0.035 [+0.011, +0.059] state contribution fell to +0.023 [−0.006, +0.051] once
`n_blocks_touched` — an acquisition counter — was moved into the progress
control (`5fb7b82`).

**N4 — Latent value can be economically unrecoverable.** In the gated family a
clairvoyant switch gained +0.153 over always-myopic; the best observable switch
captured 25%. Cause: the only regime evidence *was* the only label evidence, so
identification was paid for in the currency it was trying to save. Always-myopic
accuracy *degrades* 0.725 → 0.644 as regime-directed exploration is given more
steps (`131a57b`).

**N5 — Beating both fixed policies is not enough.** An any-time switch beat
always-myopic (+0.038) and always-strategic (+0.082) while being a **six-entry
lookup on (gate_cost, budget)**: escalation was 0% or 100% in 24/24
configurations and constant within every cell (`1e1d6e3`). The bar is the
per-cell oracle lookup, not the fixed policies.

**N6 — A cheap diagnostic is inexpressible under a scalar budget.** With integer
budgets and unit action costs, any probe price in (0,1] displaces a whole
action, so its effective price is 1.0 regardless of nominal cost. Half-integer
budgets instead produced a **budget-parity lookup capturing 94%** of oracle
decision value. Two currencies require two *resources*, not two magnitudes of
one (`b6848d8`).

**N7 — State variation in Δ\* is not the same thing as a control problem.** At
σ=0.60, B=6, `SD_s(Δ*) ≈ 0.037` is real — two independent estimators agree
(0.0345 analytic, 0.0389 empirical), and a CRN correlation test whose null is
pinned at 0.500 measures 0.776. But the mean is −0.095 and **all 40 sampled
point estimates are negative**, the best only 0.28 SE below zero.

Stated precisely: no state was *demonstrated* to have positive Δ\* at this
configuration — which is weaker than "no such state exists", since a true
Δ\* = +0.01 would routinely produce the observed −0.008. The general criterion
this yields is the useful part:

> A Governor needs `P(Δ* > 0) > 0` **and** `P(Δ* < 0) > 0` within one observable
> decision regime. Variation alone is insufficient; the distribution must cross
> the economic boundary at zero (`ff23085`, `86a87d9`).

---

## 3. The failure mode that recurred seven times

Every false positive in this project was **something invariant or privileged
wearing the costume of something learned**:

| # | Looked like | Actually was |
|---|---|---|
| 1 | a policy (+0.544) | a static feature schedule |
| 2 | cognitive state (+0.035) | task progress |
| 3 | evidence of unidentifiability | a weak hand-picked explorer |
| 4 | an any-time controller (+0.038) | a budget lookup |
| 5 | 2-step lookahead | myopic plus a constant (same choice in 26/30 states) |
| 6 | a zero-cost reflex | the strategic policy, free |
| 7 | learned cognition (+0.0227) | regime identification + config lookup |

Five were reported before being caught. Two were caught first. The difference in
every case was an **executable check on a property that had been asserted in
prose**.

### Controls that did the catching

- **Strengthen the baseline before believing the effect.** Static schedule →
  adaptive; approximate → exact; weak explorer → the VOI-optimal one.
- **Charge every control the same currency.** Deliberation that costs nothing is
  always worth doing.
- **Ask what a one-variable lookup achieves.** Below 70% of oracle decision
  value, or the "controller" is a table.
- **Compute exactly where the environment permits it.** Monte Carlo at n=32
  inverted the CUBE-NM conclusion; the estimator's own noise was being read as
  the baseline's weakness.
- **Never let a threshold be a guess.** The first likelihood gate "failed" at its
  own Bayes floor; Run A's printed verdict was wrong because a 2×-floor
  threshold missed by 0.0003.

---

## 4. Where the project stands

**Environment 5** (two resources — `B_tool` for acquisition, `B_compute` for
deliberation; three modes — reflex / diagnostic deliberation / strategic
planning) is built, instrumented and repaired. M2 is a genuine exact 2-step
lookahead (diverges from myopic in 50% of states, 96% stable at 2× quadrature
resolution). Costs are counted in four frozen primitive counters, and
`C(M0)=0 < C(M1) < C(M2)` holds in 100% of 480 measured cells.

**CLOSED.** The sign of Δ(M2−H) reverses *across* configurations — +0.267 at
σ=0.10/B=3 against −0.156 at σ=0.60/B=6 — but both ends are uniform *within*
themselves. Identical protocol, 40 states × 256 CRN draws, per-state 95% CIs:

| configuration | credibly M2 | credibly H | straddling | mean Δ\* |
|---|---|---|---|---|
| σ=0.10, B=3 | **37/40** | 0/40 | 3/40 | +0.150 |
| σ=0.60, B=6 | 0/40 | (see N7) | — | −0.095 |

Bonferroni-corrected at σ=0.10/B=3: 28 credibly M2, 0 credibly H. So one
configuration says *always escalate* and the other says *never*, and neither
contains states on both sides of zero.

**The switch is therefore a configuration-level decision, not a state-level
one** — a lookup on regime and budget, which is precisely the degeneracy gate H5
exists to reject. Environment 5 does not contain a selective-escalation problem.

**Not built:** the learned Governor. It has never been justified by a passed
construction gate, and building it before one passes is the error this
methodology exists to prevent.

---

## 5. Reproduction

```
pytest tests/ -q                              # 97 tests
python3 scripts/cube_nm_myopic.py             # the verified mechanism
python3 scripts/cube_nm_scorer_audit.py       # exact-scorer audit, 150 states
python3 scripts/gated_phase2_ladder.py        # N4, N5
python3 scripts/probe_construction_gate.py    # N6
python3 scripts/env5_gate_a_convergence.py    # N7
```

Preregistrations for Environments 4a and 5, including every recorded protocol
deviation and the road not taken, are in `PREREGISTRATION-*.md`.
