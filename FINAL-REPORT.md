# Governor — a reasoning-aware, budget-controlled cognitive layer

Final report. Every number here is traceable to an experiment directory, a
commit, and a raw file. Numbers that are not are marked as such.

**Status: Phase 4 PREMISE FAILS on this task family.** The preregistered
held-out test was not run, and running it would have been pointless: the ceiling
available to *any* allocator is at most **+0.046** at every budget (E0004). The
Env 6 result stands. See §4.4 and §8.

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
| `E0002` | preregistered primary test | NOT RUN — see §4.4 |
| `E0010` | ablations | NOT RUN — nothing to ablate against |
| `E0020-*` | robustness | NOT RUN |
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

## 9. Reproducibility

```bash
python -m pytest tests/ -q                 # 191 tests, includes frozen references

export Groq=...                            # or OR_KEY for OpenRouter
python scripts/p4_curve.py --engines qwen --n 40 --exp E0001-qwen
python scripts/p4_main.py   --engine qwen --low 700 --high 2800 \
                            --cal-items 160 --test-items 260
python scripts/p4_ablate.py --engine qwen --low 700 --high 2800 --budget 5412
python scripts/p4_robust.py --engine qwen --low 700 --high 2800 --budget 5412

python -c "from governor.harness.ledger import index; \
           [print(r) for r in index()]"    # every experiment + whether it verifies
```

Collection is resumable: responses are cached in `results/p4_cache_*.sqlite`,
and a re-run fetches only what is missing. **Commit before starting a long run**
— the ledger refuses to finalize if HEAD moves mid-run.

## 10. Provenance

Every experiment directory contains `config.json`, `results.json`,
`metrics.json`, `raw.jsonl`, `git_commit.txt`, `README.md`.
`verify_experiment(exp_id)` re-checks raw sha256, row count, nonce, config hash,
and verdict-vs-red-traps consistency, months later, from disk alone.

All commits are local. Nothing has been pushed.
