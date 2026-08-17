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

## 3a. Environment 6 — the first environment to pass the frozen gate

Per-item difficulty with a noisy observable cue. 4 items/episode, each hard with
p=0.5 INDEPENDENTLY so position carries no information; cue flipped with p=0.15
is observable, true difficulty is not. H costs 0, M2 costs 1, budget = 2 deep
calls. Designed so no constant temporal schedule can track difficulty.

    GATE 3   M1 = oracle - constant  +0.0460 [+0.0332, +0.0588]   PASS
    GATE 5   held-out seed 20260817, 800 episodes
               H 0.6897 | constant 0.7887 | greedy 0.7884
               cue 0.8131 | GOVERNOR 0.8247 | oracle 0.8387
             Governor - constant +0.0359 [+0.0262, +0.0457]
             Governor - greedy   +0.0362 [+0.0275, +0.0450]
             Governor - cue      +0.0116 [+0.0078, +0.0153]
             72% of oracle headroom; 2.00 deep calls of budget 2

**Robustness: 31/36 cells pass**, not "robust across all budgets and noise".
Five cells are INCONCLUSIVE — positive point estimates (+0.0055 to +0.0125) with
CIs spanning zero at n=500. The advantage decays monotonically as the cue
degrades (9/9, 9/9, 7/9, 6/9 at noise 0.10/0.15/0.25/0.35), which is what the
mechanism predicts; a controller still winning at 35% noise would be suspicious.

**Where the advantage comes from — ablation.** A hand-coded allocator using the
SAME opportunity-cost rule but analytic gains scores identically to the learned
Governor: +0.0000 on all three seeds, episode by episode. So the +0.0116 over
the cue heuristic is produced by the ARCHITECTURE — comparing each item against
the expected best remaining slot — and not by the model class. The learner's
contribution is that it recovers that policy from data without being given the
generative model, which is what makes the same interface portable to an LLM M2.

## 4. Where the project stands

**Environment 5** (two resources — `B_tool` for acquisition, `B_compute` for
deliberation; three modes — reflex / diagnostic deliberation / strategic
planning) is built, instrumented and repaired. M2 is a genuine exact 2-step
lookahead (diverges from myopic in 50% of states, 96% stable at 2× quadrature
resolution). Costs are counted in four frozen primitive counters, and
`C(M0)=0 < C(M1) < C(M2)` holds in 100% of 480 measured cells.

**NOT closed, and not passed — the most useful state this project has reached.**
An earlier version of this section closed Environment 5 on two *endpoint*
configurations. That was an overreach: the endpoints are where a crossing is
least likely, and testing two of twelve cells cannot settle the environment.

The middle cells — those whose aggregate Δ sat nearest zero — are where the
crossing lives. Identical protocol, 40 states × 256 CRN draws:

| configuration | mean Δ\* | SD | range | credM2 / credH / undecided |
|---|---|---|---|---|
| σ=0.35, B=4 | +0.0180 | 0.0630 | [−0.094, +0.145] | 12 / 6 / 22 |
| σ=0.60, B=4 | +0.0204 | 0.0655 | [−0.090, +0.145] | 13 / 5 / 22 |
| σ=1.50, B=4 | +0.0370 | 0.0719 | [−0.090, +0.207] | 14 / 4 / 22 |
| σ=0.10, B=3 | +0.1503 | 0.0915 | [+0.004, +0.324] | 37 / 0 / 3 |
| σ=0.60, B=6 | −0.0952 | 0.0477 | [−0.199, −0.008] | 0 / — / — |

(counts at uncorrected 95% CIs)

**Three different σ, one budget, near-identical structure.** The phenomenon
tracks the intermediate *budget*, not a regime — exactly where §1's inverted-U
predicts the transition. B=3 is deep in "always escalate", B=6 in "never", B=4
straddles.

**Under the preregistered Bonferroni correction (z=3.23) the per-state crossing
is marginal:**

| sample | credM2 | credH | verdict |
|---|---|---|---|
| seed 7 (discovery) | 3 | 0 | not MIXED |
| seed 2027 (validation) | 6 | 1 | MIXED |
| pooled, n=80, z=3.42 | 6 | 1 | MIXED *(pooling not preregistered)* |

The validation sample being *stronger* than the discovery sample argues against
overfitting, but one credibly-negative state out of forty is thin. Benjamini-
Hochberg would give a clean MIXED (13 significant, 9 pos / 4 neg); that is
recorded and **not claimed**, because Bonferroni was named in advance.

What replicates robustly is the **distribution**, not per-state significance:
mean +0.0180 vs +0.0188, SD 0.0630 vs 0.0527 across independent state samples.

**Bounded next step:** the fix is statistical power — more states or more draws
per state at σ∈{0.35,0.60,1.50}, B=4 — not another environment and not a looser
test. The most negative discovery state sits at z=−2.72 against a −3.23
threshold.

**N9 — An optimal allocation can be a constant schedule.** Env 5's binding-budget
diagnostic: signed Δ\* is −0.1250 at t=0 (SD exactly 0), −0.0119 at t=1, and
positive in only 8%/16% of later decisions with means +0.0025/+0.0050.
`P(Top2 ≠ {0,1}) = 100%` looked decisive but reduces to "never spend early" —
no state information required. **Reasoning having value and reasoning needing
adaptive allocation are different claims**; every environment here demonstrated
the first and failed the second (`c10253c`).

Stated at its real strength: under the measured decision structure there is no
evidence of a sufficiently strong, non-constant allocation signal to justify the
sequential executor. That is a termination criterion, not an impossibility proof
— the diagnostic still used local Δ\* along an H trajectory.

**Env 5 is closed for Governor development and retained as a negative control:**
it demonstrates the difference between the two claims above, which is worth more
than another tuned benchmark. `NEXT-ENVIRONMENT-GATE.md` carries the resulting
construction gate.

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
