# Governor — a reasoning-aware, budget-controlled cognitive layer

Final report. Every number here is traceable to an experiment directory, a
commit, and a raw file. Numbers that are not are marked as such.

## VERIFIED vs NOT VERIFIED  *(updated — Phase 5 PASSED)*

**The headline claim is now established, with two stated qualifications.**
On held-out items with a real LLM at an identical token budget:

| policy | U | deep calls | tokens/ep |
|---|---|---|---|
| all-cheap | 0.0625 | 0.00 | 4420 |
| heuristic | 0.1458 | 5.00 | 6146 |
| greedy | 0.1875 | 5.25 | 6147 |
| best fixed | 0.2083 | 5.75 | 6138 |
| myopic `q>0` | 0.2917 | 6.00 | 6172 |
| **GOVERNOR** | **0.3750** | 5.50 | **5643** |
| oracle | 0.4375 | 4.50 | 5392 |

`GOVERNOR − best fixed = +0.1081 [+0.0208, +0.2292]` — **58% of available
headroom captured, and it wins while spending FEWER tokens** (5643 vs 6138).
Eleven trap checks green.

**Qualification 0 — it does not replicate on external data under this resource
model.** E0013 ran the same screen on s1-32B's released MATH-500 generations at
seven budget-forced levels (third-party; I chose no item, model, prompt or
budget). The *headroom is there* — five budget pairs clear S1, best +0.1680 at
(500, 8000). **S2 fails on all 21 pairs**, best `act/cap = 0.68` against a 0.70
threshold: the model self-terminates before the cap on 98%+ of items, so a
budget that must reserve the cap reserves 2–11× what it spends. This is **not** a
Groq artifact — s1 forces `</think>` at the limit, real binding infrastructure,
and the cap is still loose, because a cap only binds the items that would have
run past it. **Hard worst-case reservation and a self-terminating model are
structurally incompatible**, and that is a property of the resource model, not
of the controller or the task.

**THREE resource contracts have now been tested on external data, and the third
passes.**

| contract | binds? | headroom? | verdict |
|---|---|---|---|
| hard worst-case reservation | no (`act/cap` 0.68) | yes (+0.168) | eliminated (E0013) |
| forced Wait units, MATH | yes (1.0) | no (+0.009) | eliminated (E0014) |
| forced Wait units, GPQA | yes (1.0) | no (+0.017) | eliminated (E0015) |
| **soft expected budget** | **yes — it IS actual consumption** | **MATH +0.170, GPQA +0.268** | **PASS (E0016)** |

`E[Σ actual generation tokens] ≤ B`, against the *strong* baseline: a fixed
policy may randomise between adjacent budget levels to match the expected budget
exactly (the upper concave envelope), not merely pick the best single level. The
oracle is the exact multiple-choice-knapsack optimum via a Lagrangian sweep.
Exact `simplescaling/s1-32B` tokenizer counts. Third-party data throughout.

**This is the contract the Governor should be built on, and it is the first one
that has both properties.**

### ⚠️ WITHDRAWN: the external MATH "PASS" was a budget violation

`E0019` reported `Governor − best fixed = +0.0282 [+0.0033, +0.0525]` **"at
identical expected budget"**. It was not.

| | utility | tokens |
|---|---|---|
| Governor | 0.8120 | **973** (budget was 846, **+15%**) |
| fixed baseline at the *nominal budget* 846 | 0.7843 | 846 |
| fixed baseline at the Governor's *realised* 973 | **0.8251** | 973 |

`G − fixed` at matched realised cost is **−0.0131**, and the matched-cost
bootstrap is **−0.0055 [−0.0540, +0.0434] — not separable**. Every E0019 variant
overspent by 109–128 tokens.

**Cause:** the Lagrangian is tuned to hit the budget on the *calibration* half;
the evaluation half costs more, so the tuned λ overspends there. Tuning on
calibration is correct. Reporting the baseline at the **nominal** budget rather
than at the Governor's **realised** cost is not.

`budget_adherence` is now trap 14: a policy may under-spend, but may not exceed
its budget, and the baseline may not be given fewer tokens than the policy used.

**So the project has NO verified claim that the Governor beats a strong fixed
baseline on external data.** The Env 6 synthetic result stands; nothing on real
LLM data does.

<details><summary>Superseded E0019 write-up (retained for the record)</summary>

### The Governor appeared to PASS on external MATH once the predictor used the right loss

**E0019** changed only the correctness predictor's loss and calibration —
same data, split, budget grid, features, allocator, baseline, contract:

| variant | mean Brier | Governor − fixed @ B* |
|---|---|---|
| ridge (what E0017 used) | 0.1031 | +0.0065 |
| logistic | 0.1026 | **+0.0318** |
| logistic + Platt | 0.1065 | +0.0118 |
| logistic + isotonic *(selected by Brier)* | **0.0975** | +0.0278 |

Held-out bootstrap at `B*=846` (chosen on **calibration** by ceiling, never by
outcome), 250 evaluation items, all traps green:

| | MATH-500 | GPQA |
|---|---|---|
| ceiling | +0.1638 | +0.2323 |
| **Governor − best fixed** | **+0.0282 [+0.0033, +0.0525] BEATS** | +0.0097 [−0.0525, +0.0505] |
| Governor − myopic | +0.0364 [−0.0021, +0.0840] | +0.0206 [+0.0000, +0.0505] |
| verdict | **PASS** | FAIL |

**This is the first PASS on third-party data**: s1-32B generations the project
did not produce, exact tokenizer costs, frozen item split, and a fixed baseline
allowed to randomise between adjacent budget levels to match the expected budget
exactly. The Governor captures **17% of the available ceiling**.

**It still does not separate from the myopic rule** (+0.0364, CI includes zero).
Whether *pricing* the resource beats merely *ranking* difficulty remains
unanswered after three attempts.

**GPQA fails, and the mechanism is known**: E0018 measured AUC ≈ 0.52 there
against 0.741 on MATH.

</details>

**Superseded:** the E0017 run below, and its diagnosis, are retained for the
record. Its verdict stood on ridge; its diagnosis was retracted in E0018.

**The learned Governor was first run under it (E0017), and FAILED on both
benchmarks:**

| benchmark | ceiling | Governor − best fixed | Governor − myopic |
|---|---|---|---|
| MATH-500 | +0.170 | +0.0069 [−0.0145, +0.0302] | +0.0239 [−0.0040, +0.0520] |
| GPQA | +0.268 | +0.0150 [−0.0815, +0.0923] | +0.0388 [−0.0205, +0.0909] |

Neither comparison separates from zero. The Governor captures roughly **4–6% of
a ceiling that is demonstrably there.**

**The bottleneck is the predictor, not the allocation rule.** Calibration CV R²
for per-level correctness runs −0.058 to +0.186 and for per-level token cost
−0.087 to +0.238. Surface text features — length, LaTeX commands, digit counts,
equation counts — do not predict *which MATH or GPQA problem will need more
tokens*. With a near-zero-signal `q̂`, the Lagrangian has nothing to price and
the myopic threshold has nothing to threshold.

This is consistent with the literature rather than surprising: Damani et al.
(2410.04707) report their online variant falling *below* best-of-k on code
because ~50% of items are unsolvable at any budget, and arXiv 2606.15841 shows
allocation headroom scales with dispersion in *signal quality*, not in
difficulty. A real ceiling and a reachable ceiling are different things.

**The two earlier contracts fail on OPPOSITE criteria** (E0013, E0014, both on s1-32B / MATH-500, exact tokenizer counts):

| contract | unit binds? | headroom? | verdict |
|---|---|---|---|
| token cap | **no** — `act/cap` 0.68, model self-terminates on 98%+ | yes, ideal +0.168 | S2 fails |
| forced Wait units (MATH) | **yes** — injections cannot be declined | **no**, ideal +0.009 | S1 fails |
| forced Wait units (GPQA) | yes | **no**, ideal +0.017 | S1 fails |

On MATH the ladder sits above the transition (0.928 at one Wait against a 0.932
saturation). **GPQA was run precisely to test whether saturation was the cause,
and it is not**: GPQA sits at 0.596, nowhere near a ceiling, and its accuracy
still *declines* 0.596 → 0.591 while tokens grow 1.7×. Forcing a model to
continue past its own stopping point does not recover accuracy on either
benchmark. Extra reasoning hurts more often than it helps in 22/28 MATH pairs
and 19/28 GPQA pairs. So the fully-consumed unit is a
clean resource that buys nothing, and the token cap is a real lever that cannot
be reserved against.

**Qualification 1 — it does not separate from budget-limited greedy**
(+0.1056 [−0.0208, +0.2089]).
**Qualification 2 — it does not separate from its own myopic ablation**
(`q>0`, no dynamic program): +0.0282 [−0.0833, +0.1458]. **The learned per-item
predictor is doing the work; the opportunity-cost DP is not demonstrably
earning its place at this sample size.** This is exactly what the literature
reports — Damani et al. (2410.04707) and arXiv 2604.14853 both measure
oracle-to-learned gaps of 0.4–1.4pp, i.e. cheap predictors capture most of the
available headroom.

Sample: 55 evaluation items forming 4 episodes. Intervals are cluster
bootstraps over ITEMS with the controller frozen.

---


**Verified.** The Governor architecture on Environment 6 (U=0.8247, +0.0359
[+0.0262, +0.0457] over the best constant schedule, 72% of oracle headroom,
frozen at 1e-12). The whole engineering stack: canonical executor, Ares
per-action layer proven trace-identical on two environments and two task
families, frozen M2 contract with four engines behind it, experiment ledger that
refuses unreproducible results, 12 executable trap checks, a 12-tool MCP harness
that provably reuses the same control loop, and a second task family with a
continuous reward running through unchanged interfaces.

**Not verified.** That the Governor helps a *real LLM* allocate a token budget.
Two task families were built for that question and both were rejected before a
controller was trained:

- **Phase 4** — the ceiling available to any allocator peaked at **+0.046**
  across every budget (E0004). Rejected.
- **Phase 4R** — structurally redesigned, in-selection ceiling **+0.1055
  [+0.0278, +0.1944]**, but the held-out gate returned **+0.0860 [+0.0000,
  +0.1667]**. The 95% lower bound is 0.0000, not above 0.02. Rejected (E0006).

No Governor was trained on either. `p4r_governor.py` refuses to run, verified.

This report is written to that outcome. The headline claim of the project —
budget-aware allocation helps a real LLM — **is not established here.**

---

## 1. Final architecture

```
            observable state                    (text only; no hidden axis)
                   |
                   v
    +--------------------------------+
    |  GOVERNOR                      |   value predictor  q(text) -> E[gain]
    |  decides, never executes       |   opportunity-cost DP over (items, budget)
    +--------------------------------+
                   |  mode = H | M2
                   v
    +--------------------------------+
    |  ARES  execute(action, state,  |   budget checked BEFORE the call
    |        budget) -> ExecResult   |   charge = measured, never nominal
    +--------------------------------+
                   |
                   v
    +--------------------------------+
    |  M2 ENGINE  (frozen contract)  |   MathM2 | Nemotron | Gemini | Qwen-Groq
    |  M2(state, budget) -> M2Result |
    +--------------------------------+
                   |
                   v
         observation + actual cost
                   |
                   +---> state update ---> repeat (4 items, one shared budget)
```

The seam that matters is `M2Result`. Four engines have been put behind it and
the Governor has never been changed to accommodate one.

## 2. Component interfaces

| component | file | contract |
|---|---|---|
| Executor (frozen) | `governor/gate/executor.py` | `run_episode(env, policy, ep, budget) -> Trace`. The only authoritative source of policy value. |
| Ares | `governor/ares/executor.py` | `execute(action, state, budget) -> ExecResult(observation, utility, consumed, state)`. Proven trace-identical to the above on two environments. |
| M2 engine | `governor/gate/m2_interface.py` | `M2(state, reasoning_budget) -> M2Result(result, reasoning_tokens, total_tokens, latency_s, cost_units, ok, error)`. |
| Environment | `governor/phase4/env.py` | `reset / observe / step / utility / modes / cap / feasible`. |
| Value predictor | `governor/phase4/predictor.py` | `q(text) -> E[gain]`; `OpportunityCostDP.threshold(items_left, k)`. |
| Ledger | `governor/harness/ledger.py` | `ExperimentRun.finalize(...)` refuses rather than records an unreproducible result. |
| Traps | `governor/harness/traps.py` | 11 checks; a red one forces `verdict=BLOCKED`. |

## 3. Experiments performed

| id | what | verdict |
|---|---|---|
| `env6-reference` (tag) | Env 6 mathematical Governor | PASS, frozen at 1e-12 |
| `E0001-qwen` | Phase 4 reasoning curve, engine + mode selection | PASS — qwen qualifies |
| E0001-nemotron | same, nemotron | NOT RUN — throttled to zero |
| `E0003-pilot` | underpowered pilot, 44 items, item-level bootstrap | PILOT — Governor −0.062 [−0.136, +0.000] vs greedy |
| `E0004-ceiling` | **what could ANY allocator gain, swept over every budget** | **PREMISE-FAILS**, max ceiling +0.046 |
| `E0008-governor-phase4r` | Governor test | **NOT RUN** — gatekeeper refuses; gate failed |
| `E0009-qwen-local` | local Qwen 1.7B-4bit backend curve (MLX) | CURVE-VALID (n=6, feasibility only) |
| `E0002` / `E0010` / `E0020-*` | primary, ablations, robustness | NOT RUN — no validated family |
| `E0005-structure` | **Phase 4R structural search**, 81 configurations, no API calls | CONFIG-FOUND — 1 of 81 passes S1+S2 |
| `E0006-ceiling-gate` | ceiling gate on that configuration | **CEILING-FAIL** — held-out CI lower bound 0.0000 |
| `E0007-structure-clean` | search re-run on the frozen selection half ONLY | CONFIG-FOUND — **identical** configuration |
| (deferred) | second task family with an LLM; local Qwen | quota-blocked |

## 4. Metrics

### Environment 6 — frozen reference (held-out seed 20260817, n=800)

| policy | U | Δ vs Governor |
|---|---|---|
| H (never deep) | 0.6896875 | |
| best constant schedule | 0.7887 | +0.0359 [+0.0262, +0.0457] |
| budget-limited greedy | 0.7884 | +0.0362 |
| observable cue heuristic | 0.813125 | +0.0116 |
| **GOVERNOR** | **0.8247** | |
| difficulty oracle | 0.83875 | 72% of headroom captured |

Robustness 31/36 cells (five inconclusive, not "robust everywhere").
Ablation: a hand-coded analytic allocator matches the learner exactly (+0.0000)
— **the architecture produces the gain, not the model class.**

### E0001 — Phase 4 reasoning curve (`qwen/qwen3.6-27b`, 40 calibration items)

| max_tokens | accuracy | starved | mean total_tokens | cap | used/cap |
|---|---|---|---|---|---|
| 300 | 0.050 | 95% | 369 | 428 | 86% |
| 700 | 0.525 | 48% | 665 | 828 | 80% |
| 1400 | 0.950 | 5% | 784 | 1528 | 51% |
| 2800 | 1.000 | 0% | 817 | 2928 | 28% |

Frozen rule → LOW=700, HIGH=2800, gap 0.475, budget 5412 (after two amendments,
§7). Raw: `experiments/E0001-qwen/raw.jsonl`, 160 rows, sha256-pinned.

### Is the gain learnable? (54 items cached at both budgets)

| observable feature | r with realised gain |
|---|---|
| `sum_numeral_log10` | **+0.718** |
| `max_numeral_log10` | +0.655 |
| `numerals` | +0.531 |
| `has_intdiv` | +0.483 |

Mean gain by hidden `n_ops`: 1 → +0.10, 2 → +0.36, 3 → +0.69, 4 → +0.89.
On 88 items the predictor reaches cv_R² = **+0.566** (spread 0.414).

### 4.4 Phase 4 — the ceiling, and why the primary test was not run (E0004)

`ceiling(B) = U(clairvoyant optimum) − U(budget-limited greedy)`, both executed
through the canonical executor, swept over every budget (88 items, 22 episodes):

| budget | all-cheap | greedy | oracle | **ceiling** | greedy deep | oracle deep |
|---|---|---|---|---|---|---|
| 4512 | 0.4318 | 0.4318 | 0.4318 | +0.0000 | 0.00 | 0.00 |
| 5312 | 0.4318 | 0.5114 | 0.5114 | +0.0000 | 0.68 | 0.32 |
| 5712 | 0.4318 | 0.7955 | 0.8409 | **+0.0455** | 2.32 | 1.41 |
| 6112 | 0.4318 | 0.8977 | 0.9205 | +0.0227 | 3.14 | 1.91 |
| 6912 | 0.4318 | 0.9773 | 0.9886 | +0.0114 | 3.86 | 2.23 |
| ≥7712 | 0.4318 | 0.9886 | 0.9886 | +0.0000 | 4.00 | 2.23 |

**Maximum over every budget: +0.046.** A perfectly-informed allocator gains
under five accuracy points, and only in a narrow band. A held-out Governor
result would have been noise around zero however good the controller was.

**Diagnosis — three compounding causes, all measured:**

1. **Supply matches demand.** The deep budget helps 47.5% of items, so about
   1.9 of 4 items per episode benefit, and the budget affords about 2 upgrades.
   Allocation only has value when you must *refuse* items that would benefit.
   Here you rarely have to.
2. **Reservation dominates choice.** The engine stops after ~817 tokens of a
   2928-token reservation (28%). Because a hard budget requires reserving the
   worst case, *feasibility* rather than *preference* decides most calls — the
   oracle can only realise 2.23 deep calls where greedy realises 4.00.
3. **The transition is narrow.** Utility goes from 0.43 to 0.99 between B=5300
   and B=6600. Outside that band every policy is pinned to the floor or the
   ceiling.

**What this does and does not say.** It does not say the architecture is wrong —
Env 6 remains a valid demonstration on a problem that *has* headroom. It says
this task family, with these two modes and a worst-case-reserved hard budget,
poses no allocation problem. The environment was eliminated for a diagnosable
reason, which is the tenth time in this project.

**What a future family would need** (a hypothesis, not something tuned into
existence here): items per episode well above the number of affordable upgrades,
so refusals are forced; and a cap-to-actual ratio near 1, so reservation does
not swamp preference. Neither is a change to the Governor.

### 4.5 Pilot (E0003), reported because it was run

44 test items, item-level cluster bootstrap, 300 resamples. Governor 0.7273 vs
greedy 0.7727: **−0.062 [−0.136, +0.000]** — not separable from zero, point
estimate negative. All ten trap checks green. Consistent with §4.4: there was
nothing to win. Recorded with verdict `PILOT` and must not be quoted as a
Phase 4 result.

### 4.6 Phase 4R — structural search (E0005)

Two criteria, **frozen before the search ran**, from the Phase 4 post-mortem:

- **S1 competition** — `P(X > K) >= 0.60` and `E[X]/K >= 1.8`, where `X` is the
  number of useful reasoning opportunities in an episode and `K` the number of
  affordable upgrades. Phase 4 had `P(X>K) = 0.275`, `E[X]/K = 0.95`.
- **S2 decidability** — `mean(actual DEEP cost) / cap(DEEP) >= 0.70`. Phase 4
  had **0.28**, so feasibility rather than preference decided most calls.

81 configurations over (items per episode, LOW, HIGH, budget), all against
cached responses. **The configuration is selected by S1 and S2; the ceiling is
reported for every cell and never used to select** — and only one cell of 81
survives both, so there was no room to choose.

The search confirms the post-mortem exactly. Every configuration satisfying S1
but failing S2 has a ceiling of **0.0000**:

| n | LOW | HIGH | act/cap | S1 | ceiling |
|---|---|---|---|---|---|
| 6 | 700 | 2800 | 0.33 | pass | +0.0000 |
| 8 | 700 | 2800 | 0.33 | pass | +0.0000 |
| 10 | 700 | 2800 | 0.33 | pass | +0.0000 |
| **6** | **300** | **700** | **0.80** | **pass** | **+0.1389** |

More items per episode does **not** create an allocation problem on its own.
Making the reservation mean something does. S2 was the binding constraint, and
an "8–12 items" rule would have missed it.

**Selected: 6 items, LOW=300, HIGH=700, budget=2868.** `P(X>K)=0.67`,
`E[X]/K=1.89`, `actual/cap=0.80`, `X=2.83` useful opportunities against
`K=1.50` affordable upgrades.

### 4.7 Phase 4R — the ceiling gate (E0006). **FAILED.**

| split | items | eps | all-cheap | best fixed | greedy | oracle | ceiling | 95% CI (item bootstrap) | gate |
|---|---|---|---|---|---|---|---|---|---|
| in-selection | 40 | 6 | 0.0556 | 0.1667 | 0.1389 | 0.2778 | +0.1389 | +0.1055 [+0.0278, +0.1944] | PASS |
| **held-out** | 24 | 4 | 0.0417 | 0.1667 | 0.1667 | 0.2083 | +0.0417 | **+0.0860 [+0.0000, +0.1667]** | **FAIL** |

The frozen criterion is a 95% bootstrap lower bound above 0.02. Held out it is
**0.0000**. Phase 4R is rejected and kept as a negative control.

**Read honestly in both directions.** The held-out point estimate (+0.0417) and
bootstrap mean (+0.0860) both exceed 0.02, so this is *not* a demonstration that
the headroom is absent — it is a **failure to establish it**. With 24 items and
4 episodes the interval is ±0.08 wide, which cannot separate "the effect shrank
from selection to held-out" from "there is not enough data to tell". Roughly 100
evaluation items — about one day of Groq quota at this configuration — would
distinguish them.

The gate was not weakened to accommodate that. A criterion that moves when it
fails is not a criterion. Confidence intervals are quoted at the item level
because episodes are groupings of a shared pool.

**The new configuration is also 2.9x cheaper**: 1256 reserved tokens per item
against 3684, i.e. 159 items/day rather than 54.

### 4.8 The split, audited and then made structural

A reviewer flagged that E0005 might have selected its configuration using items
E0006 later called held-out. **Checked against the record first**: the *selected*
configuration's S1/S2 were computed on **40 items, not 89** — only the
`(700, 2800)` pairs saw the larger pool, and those were the ones rejected. The
winner's qualification was clean.

The residual concern was real, though: evaluation items influenced which
alternatives were *rejected*, and a protocol that needs a metrics file read to
establish cleanliness is not a protocol. Three changes:

1. `configs/phase4r_split.json` freezes the split **by item id**, derived from
   the pool seed and *not* from cache state. A `pool[:40]` slice silently moves
   as collection proceeds, so the held-out set would change identity between
   runs.
2. The search is restricted to the selection half by construction.
3. **`split_leakage` is trap #12.** Overlap between the set a configuration was
   chosen on and the set it is scored on is a red check, and *missing* split
   evidence is red too — silence about a split is not evidence of a clean one.
   Adding it immediately turned a previously-green pipeline test red, which is
   the check doing its job.

**E0007 re-ran the search on the frozen selection half alone and selected the
identical configuration** — same `n=6, LOW=300, HIGH=700, B=2868`, same
`P(X>K)=0.67`, `E[X]/K=1.89`, `actual/cap=0.80`. The `(700, 2800)` rows, now on
40 items, still fail S2 at 0.28 with ceilings of exactly 0.0000. The leakage had
**zero effect on the outcome**, shown by execution rather than argued.

### 4.9 The stack, independently validated

These do not depend on any task family passing a gate, and all are tested.

**Ares** (`execute(action, state, budget) -> observation, utility, consumed,
state`). Budget checked *before* the call; a refused action does not advance
state; a loop asking for something unaffordable fails loudly rather than
silently substituting the cheap mode. Proven **trace-identical** to the frozen
`run_episode` on Env 6 and both Phase 4 families — identical actions, costs,
spend and utility — and Env 6's frozen reference utilities (0.6896875,
0.813125) are reproducible through the Ares path at 1e-12. Independence from the
controller is checked against the **import graph**, not the source text.

**MCP harness** — 12 tools over dependency-free JSON-RPC stdio:
`governor_start`, `governor_next`, `ares_execute`, `governor_status`,
`budget_status`, `m2_reason`, `graft_get_state`, `graft_update_state`,
`experiment_run`, `experiment_compare`, `experiment_index`, `gate_status`.
A test asserts the harness reproduces `run_episode` exactly, so the plugin is
not a second control loop that can drift. `graft_update_state` writes to an
isolated scratch dict the allocator never reads, and a test proves a written
hint cannot change a decision — Env 5 manufactured a +0.035 "cognitive" effect
from a progress counter, and a writable memory the policy consumed would let
that recur through a tool call. Every invocation is recorded with tool, args,
latency, commit and error, **including calls that raised**. The gatekeeper
applies: `governor_start("phase4r")` raises.

**Second task family** — constraint puzzles: different difficulty cue
(constraint count and type, not numeral magnitude), different encoding
(structural counts), different reward (**fraction of seats correct**, so gains
are continuous rather than in {-1,0,+1}). Every puzzle is re-solved from its
prompt text at test time rather than trusted from the generator, and a test
asserts that puzzles sharing a feature vector can have different answers, so the
observable encoding is not a channel for the solution.

**This family found three real defects** that no amount of review would have:
`ValuePredictor` and `P4Env` imported family one's feature extractor directly,
and `calibrate()` iterated a hardcoded arithmetic feature tuple. All three
worked perfectly and all three were silently arithmetic-only. `Family` is now a
first-class object and the generalization test passes by changing **one
argument**.

**Graft** — the cognitive-state ablation harness exists and is unit-tested
(components: text / progress / budget / history / uncertainty, with a bootstrap
ensemble for the last). Its *prediction is recorded in the module docstring
before any run*: only text should matter to the predictor, because an item's
gain does not depend on where in the episode it appears or on what earlier calls
cost. **The scientific ablation was not run**, because "keep a component only if
held-out utility improves" needs a family with validated held-out headroom, and
neither family has one. Running it on synthetic data would manufacture an answer.

**Local Qwen backend** (`Qwen3-1.7B-4bit` via MLX; 1.7B chosen from measured
hardware — 8 GB unified memory, ~9 GB free disk — not from what would look
best). Fourth engine behind the unchanged M2 contract. Measured finding: under
the terse system prompt the model emits an **empty** `<think></think>` and
answers in 11 tokens, so a token budget cannot matter to it — the hosted
`qwen3.6-27b` reasons under the identical prompt. See §4.10.

### 4.10 Local Qwen backend (E0009)

`mlx-community/Qwen3-1.7B-4bit` via MLX. Size chosen from measured hardware —
8 GB unified memory, ~9 GB free disk — not from what would look best. Fourth
engine behind the unchanged M2 contract.

| max_tokens | accuracy | starved | mean total_tokens | s/call |
|---|---|---|---|---|
| 300 | 0.000 | 83% | 316 | 10.1 |
| 700 | 0.500 | 33% | 577 | 21.8 |
| 1400 | 0.500 | 33% | 810 | 28.5 |
| 2800 | 0.667 | 17% | 1180 | 47.7 |

The frozen mode-selection rule qualifies it: LOW=1400, HIGH=2800, gap 0.167.
**n=6 items**, so each accuracy carries roughly ±0.2 — this is a backend
feasibility measurement, not a competence claim.

What it establishes: the M2 seam holds for an engine with completely different
runtime characteristics — no network, no per-day token quota, and throughput
bounded by local memory bandwidth (10–48 s/call) rather than someone's rate
limiter. The Governor was not modified.

One measured oddity worth recording: under a terse "reply with only the integer"
system prompt with `enable_thinking=True`, this model emits an **empty**
`<think></think>` and answers in 11 tokens. The hosted `qwen3.6-27b` reasons
under the identical prompt. A reasoning-budget experiment on a model that has
been talked out of reasoning would be vacuous, and that is a per-engine
property worth checking before designing around one.


## 5. Budgets, models, runtime

- Engine: `qwen/qwen3.6-27b` via Groq, `temperature=0`, budget lever
  `max_completion_tokens`.
- Charged cost: the provider's `usage.total_tokens`. Never nominal, never
  `reasoning_tokens` alone, never call count.
- Episode budget 5412 tokens for 4 items; feasibility reserves
  `cap(mode) = 128 + max_tokens`.
- Runtime fingerprint (python, numpy, sklearn, platform) is recorded in every
  `experiments/*/config.json`.

## 6. Baselines and ablations

Baselines, all executed through the canonical executor: all-cheap, best fixed
schedule (frozen on calibration), all-deep, budget-limited greedy, a
single-feature observable text heuristic (feature and threshold frozen on
calibration), and an enumerated clairvoyant optimum.

Planned ablations (E0010): DP vs fixed threshold vs `q>0`; gbt vs ridge vs a
mean null model; cognitive state text / +progress / +budget / +history /
+uncertainty. **The prediction is recorded before running**: only text should
matter to the predictor, because an item's gain does not depend on where in the
episode it appears or on what earlier calls cost.

## 7. Failure and retraction history

Sixteen recorded corrections across the project. Every one was caught by an
executable check on a claim asserted in prose — never by a better model, never
by more data. This session added:

| # | what went wrong | how it was caught |
|---|---|---|
| 17 | Operator precedence made the expression rendering of a chain mean something different from the word rendering — a silent 50% mislabelling of every 3-and-4-op item | a test that recomputed every answer from the printed expression |
| 18 | The `// is integer division` note appeared on every expression item, turning `has_intdiv` into a framing detector | inspection of the feature definition against the prompt |
| 19 | The oracle was capped at the worst-case upgrade count while other policies got extra slack from under-spend, so the "ceiling" lost to a fixed schedule | a test asserting the oracle bounds every implementable policy |
| 20 | `frozen_before_heldout` compared the run's commit to itself and was therefore always red | the trap firing on a healthy run |
| 21 | The ledger's dirty-check counted the experiment's own output as code drift, and parsed status paths by column offset after the output had been stripped | every experiment failing to finalize |
| 22 | **A cache-key change silently invalidated 262 paid-for responses.** A fresh process would have re-collected all 1040 calls against a hard cap of 1000/day | a plumbing dry-run that found 0 cached items where it expected 88 |
| 23 | **Amendment-1 budget made position 0 unupgradable** — the Env 5 temporal confound, silently reacquired | measuring per-position feasibility instead of assuming it |

Two claims were retracted before they became results:

- *"The gain is not predictable from text"* — a 40-item smoke test showed
  cv_R² = −0.062. Wrong: it was gradient boosting overfitting 40 samples, and a
  single feature correlates +0.718.
- *The preregistered budget formula* — it produced a budget under which greedy
  upgrades all four items, so nothing would have been measured.

### Voided, not silently discarded

The Gemini reasoning curve is **VOID**: 492/500 calls returned HTTP 429. Its
measured semantics (`thinkingBudget` is genuinely enforced, `thoughtsTokenCount`
is observable) stand; its accuracy numbers do not, and must not be read as a
weak-model result.

## 8. Known limitations

1. **The Phase 4 held-out test was not run, and should not be.** The ceiling
   measurement (E0004) shows no allocator could gain more than +0.046 at any
   budget on this task family. No claim is made about
   `U(Governor + LLM) > U(best fixed policy)` on real LLM items.
2. **Quota, measured the hard way.** Three limits exist and the binding one was
   named only in a 429 body: Groq TPM 8000, RPD 1000, and **TPD 200000 charged
   on RESERVED max_completion_tokens** — about 54 items/day, so the 420-item run
   needs eight days. OpenRouter's free tier is 50 requests/day against a
   requirement of ~1400. The directive named nemotron as
   the Phase 4 engine; the frozen selection rule fell through to qwen because a
   throttled engine cannot qualify. $10 of credits would raise the cap to
   1000/day and make the comparison possible.
3. **Test set is 65 episodes**, sized to the 1000-request daily cap. The 95% CI
   half-width will be roughly ±0.06, so effects below that are undetectable.
4. **One engine, one task family, measured.** The second family (constraint
   puzzles with partial credit) is built and unit-tested but has no LLM data.
   The local Qwen backend is untested: this machine has 8 GB of unified memory
   and 7.3 GB of free disk, so Qwen3-1.7B-4bit is the realistic ceiling, not 4B.
5. **Responses are cached and shared across policies.** This is common random
   numbers, applied identically to every policy including the oracle, and it is
   why seven policies fit inside the rate limits. It means the comparison is
   matched but is not a test of run-to-run variance.
6. **The cap/actual gap is large.** The engine stops after ~817 tokens of a
   2928-token reservation, so policies reserve about 3.6x what they spend.
   Budget utilisation is reported rather than assumed away, and the budget is a
   swept dimension in the robustness run.
7. **Env 6's Governor is validated; Phase 4's is not yet.** The architecture
   claim rests on Env 6 until E0002 reports.

## 11. Generalization results

| axis | result |
|---|---|
| second task family (continuous reward) | interfaces unchanged; **found 3 hardcoding defects** |
| second environment (Env 6) for Ares | trace-identical, frozen values at 1e-12 |
| fourth engine behind M2 | local MLX backend, Governor unmodified |
| MCP harness vs test harness | proven identical execution |

## 12. Trap checks

Twelve executable checks; a red one forces `verdict=BLOCKED` in the ledger and
missing evidence is red, not silent: greedy collapse, constant schedule, oracle
leakage, answered-vs-utility, token accounting, execution-vs-scoring,
progress-as-cognition, MC convergence, invariant-as-intelligence,
frozen-before-heldout, **split leakage**, secret scan.

They fired for real four times this session: on the degenerate smoke fixture
(three at once), and `frozen_before_heldout` on its own self-comparison bug.

## 9. Reproducibility

See `REPRODUCE.md`. Everything below runs with **no API key**:

```bash
make test      # full regression suite
make smoke     # end-to-end: both families, Ares, MCP, traps, ledger
make verify    # re-verify every recorded experiment from disk
make mcp       # MCP server over real stdio JSON-RPC
```

Needing credentials:

```bash
export Groq=...
make collect   # fills evaluation items as the per-day bucket refills
make gate      # Phase 4R held-out ceiling gate
make governor  # refuses unless the gate recorded CEILING-PASS
```

**The single highest-value next experiment**: collect ~100 Phase 4R
evaluation items (about one day of Groq quota) and rerun `make gate`.
That is what separates "the effect shrank from selection to held-out"
from "24 items could not tell".

Collection is resumable: responses are cached in `results/p4_cache_*.sqlite`,
and a re-run fetches only what is missing. **Commit before starting a long run**
— the ledger refuses to finalize if HEAD moves mid-run.

## 10. Provenance

Every experiment directory contains `config.json`, `results.json`,
`metrics.json`, `raw.jsonl`, `git_commit.txt`, `README.md`.
`verify_experiment(exp_id)` re-checks raw sha256, row count, nonce, config hash,
and verdict-vs-red-traps consistency, months later, from disk alone.

All commits are local. Nothing has been pushed.
