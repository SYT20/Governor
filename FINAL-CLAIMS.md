# FINAL CLAIMS

Every row is backed by a recorded experiment that re-verifies from disk
(`make verify`). A positive point estimate is never reported as VERIFIED.

## VERIFIED

| claim | evidence | experiment | limitation |
|---|---|---|---|
| The allocation architecture beats the best constant schedule in a synthetic environment with real headroom | U 0.8247 vs 0.7887, Δ +0.0359 [+0.0262, +0.0457], 72% of oracle headroom, pinned at 1e-12 | `env6-reference` | Synthetic; difficulty is an injected latent bit |
| A closed-form law predicts the adaptive ceiling before any controller is built | `ceiling(n,k,p) = (E[min(k,X)] − k·p)/n`; matches simulation to 4e-3; explains both prior family rejections with different causes (39.5% vs 91.2% realisation) | `headroom.py`, E0004–E0007 | Binary gains; continuous case by simulation |
| The soft expected-budget contract has material adaptive headroom on external data | MATH +0.170, GPQA +0.268 against a randomised-envelope fixed baseline and a multiple-choice-knapsack oracle | E0016 | Ceiling only — says nothing about reachability |
| Observable features carry allocation signal on external MATH | AUC 0.671 question-only, 0.741 with a 500-token probe | E0018 | GPQA ≈ 0.52, no usable signal |
| The predictor's loss materially changes allocation | ridge +0.0065 → logistic +0.0318 on identical inputs | E0019 | Both measured before budget enforcement |
| Ares is trace-identical to the frozen executor | identical actions, costs, spend, utility on two environments and two task families; Env 6 reference reproduced at 1e-12 | `tests/test_ares.py` | — |
| The MCP harness reuses the same control loop | test asserts it reproduces `run_episode` exactly | `tests/test_mcp.py` | — |
| The architecture is task-family independent | second family (constraint puzzles, continuous reward) runs through unchanged interfaces by passing one argument | `tests/test_second_family_e2e.py` | Synthetic responses |
| A local MLX backend fits the frozen M2 contract | Qwen3-1.7B-4bit, curve qualifies under the frozen rule | E0009 | n=6; feasibility, not competence |

## NOT VERIFIED

| claim | status | evidence |
|---|---|---|
| **The Governor beats a strong fixed policy on real LLM data** | **UNRESOLVED across 4 experiments** | MATH tokens +0.0121 [−0.0396, +0.0510] (E0021); GPQA +0.0000 (E0021); LCB samples −0.0028 [−0.0066, +0.0000] (E0024); LCB samples + probe +0.0046 [−0.0047, +0.0171] (E0025) |
| Early-generation signal recovers the allocation gain | **INCONCLUSIVE** — but directionally positive: −0.0028 → +0.0046, and 0-0 → 2-0 on discordant outcomes (E0024 → E0025) |
| The Governor beats a fixed policy on GPQA | INCONCLUSIVE | +0.0000 — enforcement collapses all policies to one allocation |
| Opportunity-cost pricing beats difficulty ranking | **UNRESOLVED after 4 attempts** | +0.0202 [−0.0200, +0.0600]; McNemar p=0.42 on 101 disagreements (E0020, E0021) |
| A probe pays for itself | NOT TESTED | At B*=846 a 500-token probe is 59% of the budget; untestable at this operating point |

## WITHDRAWN

| claim | why |
|---|---|
| `v1.8` "Governor beats best fixed on external MATH, +0.0282 [+0.0033, +0.0525]" | The Governor spent 973 tokens against a budget of 846 (+15%) and was scored against a baseline at the *nominal* budget. At matched realised cost the sign flipped to −0.0131. Trap 14 now blocks it. |
| E0017's diagnosis "features carry no signal" | Ridge R² on a sparse binary target; logistic AUC on the same data is 0.613–0.741. The *result* stood; the *diagnosis* did not. |
| Gemini reasoning curve | 492/500 HTTP 429. VOID, not a weak-model result. |

## ELIMINATED RESOURCE CONTRACTS

| contract | binds? | headroom? |
|---|---|---|
| hard worst-case reservation | no — `act/cap` 0.28–0.68 | yes, +0.168 |
| forced Wait units (MATH, GPQA) | yes | no, +0.009 / +0.017 |
| **soft expected budget + hard runtime cap** | **yes** | **yes** — the one in use |

## Third external axis: LiveCodeBench sample allocation (E0023, E0024)

MATH-500 was closed as unsettleable, so the **allocation axis** was changed
rather than the benchmark enlarged. LiveCodeBench's published submissions carry
pass/fail for 10 independent samples on 400 problems with the raw generations —
a complete multi-budget table at zero API cost.

**Ceiling PASSES** (E0023): +0.0573 at E[tokens]=219, ~2.9x the threshold.

**Governor does not capture it** (E0024), all traps green, budget adherent
(−1.5%), architecture reused unchanged:

| policy | U | tokens | mean k |
|---|---|---|---|
| GOVERNOR | 0.3737 | 151 | 1.23 |
| myopic | 0.3737 | 153 | 1.14 |
| fixed @ matched cost | 0.3765 | 151 | |
| oracle | **0.4316** | 151 | |

`GOV − fixed = −0.0028 [−0.0066, +0.0000]`; `GOV − myopic = +0.0000`.

**The diagnostic is the striking part: 45 allocation disagreements produced ZERO
outcome differences.** The two policies chose different sample counts on 45
problems and the pass/fail outcome was identical on every one. Given the
structure — 54.5% never pass at any k, 30.5% pass at k=1 — the controller is
reallocating among problems where reallocation *cannot* change the result. This
is precisely the failure Damani et al. report on code, where an online allocator
falls below uniform because it cannot identify unsolvable items.

The ceiling (+0.055 here) lives almost entirely in the 15% mixed problems, and
the text-only features do not find them.

## FINAL EXPERIMENT: early-generation signal (E0025) — INCONCLUSIVE, hard stop

The diagnosed problem was that problem text does not locate the items worth
spending on. The final mechanism tested was the one the evidence pointed at:
give every problem a **2-sample probe, always paid for**, and read the signal
off those generations — length, whether a code block parsed, program size, and
cross-sample **agreement** (do the two independent samples produce the same
program). Sample correctness never enters a feature.

| policy | U | tokens | mean k |
|---|---|---|---|
| GOVERNOR | 0.4000 | 294 | 2.39 |
| myopic | 0.3895 | 271 | 2.05 |
| fixed @ matched cost | 0.3952 | 294 | |
| oracle | **0.4526** | 294 | ceiling **+0.0574** |

`GOV − fixed = +0.0046 [−0.0047, +0.0171]` — **not separable**.
`GOV − myopic = +0.0104 [+0.0000, +0.0263]`.

**The probe did move the needle in the predicted direction.** Against E0024's
text-only Governor (−0.0028, and 45 disagreements producing *zero* outcome
differences), the probe version reaches +0.0046 with 51 disagreements producing
**2 wins and 0 losses**. Directionally consistent with the hypothesis; nowhere
near significant.

**One trap is RED**: `budget_adherence` — the Governor spent 294 against a
nominal B*=279 (+5.2%), because the mean-cost reserve is not a hard guarantee.
The *primary comparison is unaffected*: the fixed baseline is scored at the
Governor's own realised 294 tokens, so the contest is matched. But by this
project's own rule a red trap blocks a PASS, and the verdict is INCONCLUSIVE
regardless of it.

**HARD STOP.** This was the final proposed mechanism. The allocation claim is
recorded as unresolved. No further benchmarks.

## Why the real-LLM claim cannot be settled here (E0022)

Power computed from the **observed per-item paired variance**, not from a CI
half-width:

| comparison | n | mean | sd | items required for CI lower bound > 0 |
|---|---|---|---|---|
| Governor − myopic, all items | 250 | +0.0040 | 0.3293 | **26,031** |
| Governor − myopic, disagreements only | 107 | +0.0093 | 0.5046 | 11,199 |

**MATH-500 contains 500 items.** The required sample is ~52× the entire
benchmark. The claim can be neither established nor refuted on this dataset.

This is structural, not a compute or tooling limit. Most items produce identical
allocations (d = 0) while disagreements contribute ±1, so the mean is small over
a large standard deviation — the worst ratio for detection.

*Correction:* I previously estimated 2000–3000 items. That was wrong by ~10×
because it scaled from a bootstrap CI half-width instead of the per-item paired
variance. The correct figure is above.

## Reproduce

```bash
make test          # 235 tests
make smoke         # end-to-end, both families, MCP, traps, ledger
make verify        # re-verify every experiment from disk
python scripts/enforced_governor.py --bench math
```
