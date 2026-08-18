# Governor — Architecture

A reasoning-aware, budget-controlled compute-allocation layer for LLM agents.

Governor decides **where scarce reasoning compute is worth spending**. It does
not reason, and it does not execute. Those are separate layers behind frozen
interfaces, which is what lets four different reasoning engines sit behind one
contract without the controller knowing which one answered.

---

## 1. System architecture

![Governor harness — end-to-end architecture](docs/architecture.png)

An agent or host calls in over MCP. The Governor reads the current task and
agent state, decides how much reasoning budget this step is worth, and hands
that decision to the layers that carry it out. Results and their measured cost
flow back, the state advances, and the loop repeats until the task ends or the
budget does.

| Component | Responsibility | Why it is separate |
|---|---|---|
| **Agent / Host** | issues the task and the budget | Governor is a library, not an agent — it never decides *what* to do, only how much to spend |
| **MCP Server** | 12 tools over dependency-free JSON-RPC stdio | any MCP host drives the same control loop, so there is no second implementation to drift |
| **Governor** | adaptive compute allocation | compares predicted gain against opportunity cost — what those tokens would buy on the steps still to come |
| **State Manager** | task and agent state | holds only what is observable at decision time; correctness is deliberately absent because nothing knows it yet |
| **Action Executor** | tools and environment actions | tests the call against the remaining budget **before** dispatch, so a refusal costs nothing and advances no state |
| **Reasoning Engine** | the LLM itself | four interchangeable backends behind one result contract |
| **External Evaluator** | benchmark and tests | scoring is outside the loop, so the controller can never see its own grade |
| **Experiment Ledger** | provenance and validation | refuses to record a result it cannot reproduce |

**The rule that holds the design together:** there is a *single* execution path.
`tests/test_reference_frozen.py` pins the canonical episode loop at 1e-12, and
further tests assert that the per-action execution layer and the MCP harness
both reproduce it byte-for-byte on two environments and two task families. A
plugin with its own control loop would drift from the validated one while both
kept appearing to work.

---

## 2. End-to-end workflow

![Governor harness — end-to-end workflow](docs/workflow.png)

One pass of the loop, from task arrival to returned result.

1. **Task received.** The host supplies the task and a token budget.
2. **Read current state.** Observable features only — prompt statistics, budget
   remaining, and how prior steps behaved.
3. **Compute Governor.** A calibrated predictor scores the expected gain of
   spending more on this step.
4. **Allocation decision.** Either the step justifies reasoning, or the budget
   is exhausted and the run stops early. The budget test happens *here*, before
   any spend.
5. **Run reasoning engine → execute actions.** The chosen effort level is
   dispatched, and tools or environment actions are carried out.
6. **Record outcome and resource usage.** Cost is charged from an exact
   tokenizer — never a nominal figure and never an estimate.
7. **State transition.** Effort and cost are written back; correctness is not,
   because at this point it remains unknown.
8. **Record provenance and validation.** Telemetry lands in the ledger, which
   verifies commit, hashes and budget adherence before a verdict stands.

The loop closes at step 7: because cost is *measured*, the next allocation
decision sees a truthful remaining budget.

---

## 3. Experiment protocol — ceiling before controller

The order is frozen. Reversing it is how this project once burned a day
building a controller for an environment that had no headroom to recover.

Four gates run in sequence, each able to end the investigation outright. The
first three are computable on cached data in microseconds; only the last costs
real quota, which is why it runs last.

| Gate | Criterion | What failing it means |
|---|---|---|
| **S1 — headroom** | closed-form ceiling ≥ 0.12 | no allocator, however good, could gain anything here |
| **S2 — decidability** | mean actual ÷ cap ≥ 0.70 | the reservation dominates the preference — you would be rationing something that was never scarce |
| **S3 — stability** | k-drift ≤ 0.30, ceiling loss ≤ 20% | the operating point moves between splits, so a setting tuned on one is wrong on the next |
| **Ceiling gate** | oracle − greedy > 0.02, bootstrap CI lower bound clearing it, held-out items only | the prize is not there, or not large enough to survive its own error bars |

`require_gate_passed()` raises rather than returning a flag: nothing downstream
of the gate may run until a `CEILING-PASS` is on record. Rejected families are
retained as negative controls, never deleted.

Only then: fit the predictor on the calibration split, build the allocation
rule, and evaluate on held-out items **at matched realised cost** — not at the
nominal budget.

---

## 4. Resource contract

A generation's cost is unknown until after it has been produced, so "a budget"
has to be defined before the spend is knowable. Three definitions were
implemented and measured on external benchmarks. Two were eliminated.

| Contract | Measured outcome | Verdict |
|---|---|---|
| **Hard worst-case reservation** — reserve the cap for every call | engines used 28% of what was reserved | **rejected** — admission decided by what would *fit*, not what was *worth it* |
| **Forced consumption** — compel N extra reasoning units | budget binds exactly; ceiling +0.009 | **rejected** — tight control over a dial with no coupling to utility |
| **Soft expected budget** — `E[Σ actual tokens] ≤ B`, plus a hard runtime cap | binds and retains headroom | **adopted** |

The adopted contract is enforced, not documented: `budget_adherence` fails any
run whose realised cost exceeds its budget by more than 2%, and fails any
comparison in which the baseline was given fewer tokens than the policy spent.
Under-spending is permitted — a policy that wins while spending less is a real
win.

---

## 5. Frozen interfaces

| Layer | Contract | Frozen by |
|---|---|---|
| Episode loop | `run_episode(env, policy, ep, budget) -> Trace` | reference test at 1e-12 |
| Per-action execution | `execute(action, state, budget) -> ExecResult` | trace-identical to the episode loop, two environments |
| Reasoning engine | `M2(state, reasoning_budget) -> M2Result` | four backends, Governor unchanged |
| Environment | `reset / observe / step / utility / modes / cap / feasible` | `observe` carries no correctness and no future |
| Task family | `features, feature_names, grade, system_prompt, heuristic_features` | swapping it is the *only* change needed for a new task family |
| Ledger | `ExperimentRun.finalize(...)` refuses rather than records | 13 provenance tests |

---

## 6. Governance — 15 executable traps

Every check in `governor/harness/traps.py` exists because a specific failure
already happened and printed "PASS" first. Missing evidence is a **red** check,
so silence never reads as success, and a red check forces `verdict = BLOCKED`
at the ledger regardless of what the caller reported.

Representative examples:

- `oracle_leakage` — fails on feature *names* if anything resembling a label,
  difficulty or future value appears in the state.
- `exact_token_counts` — refuses estimated costs. Costing generations at
  `len(text)/4` was off by 25–35%, and the error moved with the budget level.
- `budget_adherence` — a policy compared at a budget it did not respect is not
  a comparison.
- `split_leakage` — selection and evaluation item ids must be disjoint.
- `withdrawn_result_promotion` — the ledger is append-only, so a withdrawn
  result still reads PASS on disk. Preserving the row is right; citing it is not.

See `TRAPS.md` for the full inventory and which ones have actually fired.

---

## 7. Status

| Axis | State |
|---|---|
| **ENGINEERING** | GREEN — 257 tests, 26/26 experiments verify, smoke passes |
| **SCIENCE** | real-LLM advantage **NOT VERIFIED** — 4 experiments, 3 axes, every CI crossing zero |

These are different axes, and a green one does not imply the other. The
measured ceiling is real every time — +0.164 on MATH, +0.232 on GPQA, +0.055 on
LiveCodeBench — but observable features never locate the items where spending
pays. The sharpest diagnostic: in E0024, **45 allocation disagreements produced
zero outcome differences**. The controller reallocates among items where
reallocation cannot change the result.

That is a claim about observability, and it is reported as a negative result
rather than buried. See `FINAL-CLAIMS.md` for the full VERIFIED / NOT VERIFIED /
WITHDRAWN matrix.
