# Governor

An external, empirically-calibrated controller that decides how an LLM agent spends a
limited budget *within* a single episode — what to do next, at what effort, and when to
stop — using measured costs and calibrated probabilities instead of the model's own
estimate of how it is doing.

**Status: Stage 1 of 9.** The control machinery runs end-to-end on a synthetic
environment. It has never touched a real repository. See [Limitations](#limitations)
before reading any number below.

---

## Positioning

This is not a novelty claim. Adaptive budgeted compute allocation for LLMs already
exists. Governor sits in a specific gap between three published lines of work:

- **[BAGEN](https://arxiv.org/html/2606.00198)** measures whether agents know their own
  remaining budget, and finds they do not: systematic optimism across all 20
  model-environment pairs tested, still predicting "feasible" >70% of the time after
  60% of budget is gone, and interval coverage capping at 47% even after SFT+RL. It
  builds no controller and explicitly names online estimation as future work.
- **[AVA](https://openreview.net/forum?id=JMDCMf7mlF)** builds a budget-constrained
  controller with calibrated uncertainty — but on GSM8K, MATH, HotpotQA and HumanEval.
  No repository state, no tool-use loop, no deterministic regression tests.
- **[The Replay Gap](https://arxiv.org/html/2608.08239)** shows that evaluating
  mid-trajectory policy changes by replaying logs is invalid: 92–97% of decisions get
  scored against states that never occurred. It provides a branching protocol but no
  policy.

Governor is the intersection: a controller for the agentic repo-level domain, fit on
deliberately randomised rollouts so action effects are identifiable, and evaluated by
live branching rather than replay.

## What is actually built

| Component | Status | Guards |
|---|---|---|
| `core/estimate.py` — provenance-carrying numbers | working | A bare `float` cannot enter the policy. `fitted` values must carry an interval, a model id, and a corpus version |
| `accounting/meter.py` — metering **and** hard enforcement | working | BVR = 0 is structural, not statistical. An overrun raises; it is never logged |
| `cognitive/belief.py` — two-parameter evidence channel | working | Corrected asymmetric likelihood (α, β). Beta-Binomial estimation with a conservative fallback |
| `envs/synthbug.py` — synthetic environment, known ground truth | working | Deterministic per seed; exact oracle available |
| `models/cost.py` — empirical cost quantiles | working | Frozen before evaluation; refuses to learn mid-run |
| `policy/runner.py` — episode loop, dual-source candidates | working | Same code path for every arm (arm parity) |
| `arms/baselines.py` — A/B/C, ε-greedy collector, oracle | working | — |
| `record/store.py` — append-only decision records | working | Every episode reconstructible without re-running |
| Value model, Governor (arm E), branching harness | **not built** | Stages 3, 5, 7 |
| Real SWE-bench executor | **not built** | Stage 2+, blocked on hardware |

## Quickstart

No API key, no GPU, no Docker, no third-party packages. Python 3.11+ only.

```bash
make test    # 33 invariant tests, ~40ms
make smoke   # small end-to-end sweep, ~30s
make run     # full sweep: 3000 episodes, ~3min
```

## Current results — machinery only

3000 episodes on SynthBug, 4 hypotheses, 5 budget levels.

```
 budget arm                TSR  cost/win   BVR  trunc
  100% A_fixed          38.3%    0.1629    0%   0.00
  100% C_heuristic      81.7%    0.1747    0%   0.00
  100% F_oracle        100.0%    0.0675    0%   0.00
   25% A_fixed          28.3%    0.1639    0%   0.03
   25% C_heuristic      21.7%    0.2329    0%   0.05
   25% F_oracle         70.8%    0.0463    0%   0.02
```

Three things this establishes:

1. **BVR = 0 across every arm and every budget level.** The enforcement layer holds
   under adversarial charging, which is what makes the compliance claim an engineering
   property rather than a statistical hope.
2. **The channel estimator recovers the parameters that generated the data** to within
   0.019 absolute across all three tiers. The corrected two-parameter Bayes update is
   arithmetically right and its estimator is unbiased.
3. **Headroom over the best baseline more than doubles as budget tightens: +18.3% at
   full budget → +42.5% at 25%.** This is the prediction the whole project rests on —
   that better allocation matters most under scarcity — and it is now measured rather
   than assumed. Note the hand-tuned heuristic *collapses* under scarcity, falling
   below even the fixed baseline; its budget-floor rule bails too early.

At 10% budget every arm including the oracle scores 0%. That level is below the floor
where any policy can act and is excluded from claims.

## Limitations

Read this before citing anything above.

- **SynthBug is a simulator.** It validates accounting, belief updating, admissibility,
  the arm harness, and the metrics pipeline. It says **nothing** about real code-fixing
  skill. No result here transfers to SWE-bench without being re-measured there.
- **Governor itself does not exist yet.** The arms compared are a fixed script, a static
  router, a hand-tuned heuristic, an ε-greedy collector, and a cheating oracle. The
  calibrated controller is Stage 5.
- **The oracle is a ceiling, not a competitor.** It reads the hidden true cause.
- **No statistical testing yet.** Single seed set, no confidence intervals, no paired
  analysis. The numbers are directional.
- **The hardware forced two substitutions** from the design: SQLite instead of
  Parquet/DuckDB, and a synthetic environment instead of Docker + SWE-bench. Both are
  one-module swaps, but they are swaps.

## Design

The full decision record — hypotheses, falsifiers, the deferred decisions and the
staged plan — is in [`Governor-Decision-Record.md`](Governor-Decision-Record.md).

Two principles drive the code:

**Every number carries provenance.** `Estimate` requires a declared source
(`measured` / `fitted` / `derived`), and `fitted` additionally requires an interval, a
model id, and a corpus version. The policy refuses anything else. This makes the
project's central commitment — the LLM does not invent the numbers that govern
spending — a type error rather than a convention.

**Estimation and enforcement are separate systems.** Statistical admissibility (p90
cost + a state-dependent reserve) reduces how often actions overrun; it guarantees
nothing, and its metric is the truncation rate. Runtime enforcement guarantees BVR = 0
structurally by handing the executor a hard cap. Conflating these is how projects claim
budget compliance they cannot deliver.

## Roadmap

| Stage | Deliverable | Gate |
|---|---|---|
| 0–1 | Skeleton, accounting, enforcement, environment | **done** — BVR = 0 |
| 2 | ε-greedy corpus, feature extractor | Effective sample size ≥ 200, action coverage complete |
| 3 | Value model + calibration | **Beat a base-rate predictor on ECE and Brier**, or stop |
| 4 | Policy v0, arms A/B/C on real data | BVR still 0 |
| 5 | Cognitive state, arms E and E⁻ᶜᵒᵍ | Does explicit belief state earn its place? If it ties, cut it |
| 6 | LLM-as-controller arm, degradation curves | Separation visible at 25% |
| 7 | Branching harness, sampled-branch regret | Control-fork fidelity verified |
| 8 | Robustness, ablations, transfer | — |
| 9 | Release | A stranger reproduces from the README alone |

## Attribution

Planned upstream dependencies (not yet integrated): SWE-bench harness (MIT), SWE-Gym,
SWE-smith, mini-swe-agent (MIT), vLLM (Apache-2.0), Qwen3 (Apache-2.0). Licences will be
audited and recorded in `NOTICE` at integration time.

Papers cited above are the intellectual context, not dependencies.

## Licence

TBD before first release.

## How to resume this project

```bash
python scripts/resume_project.py   # orientation in under a minute; starts nothing
python scripts/project_health.py   # ENGINEERING_HEALTH + SCIENTIFIC_STATUS
make test && make verify && make smoke
```

| | |
|---|---|
| checkpoint tag | `v2.2-final` (engineering), `v2.2-robustness` (hardening) |
| commit | see `scripts/resume_project.py` — do not trust a remembered hash |
| memory | `governor-v2.2-checkpoint.md`, indexed first |

**Do not resume by memory alone. Re-verify the repository, tag, tests, ledger
and final claims.** Memory records what was true when it was written.

### Read these as different axes

| axis | state |
|---|---|
| **SOFTWARE VERIFIED** | canonical executor, Governor, Graft, M2 contract with four backends, Ares, MCP harness, ledger, 15 traps, exact-token accounting, two task families, reproducibility tooling |
| **SCIENCE VERIFIED** | Env 6 synthetic result; the closed-form headroom law; soft expected budget as the only viable resource contract of three; observable signal on MATH (AUC 0.741); predictor loss materially changes allocation |
| **SCIENCE UNRESOLVED** | the Governor beating a strong fixed policy on real LLM data — four experiments, three axes, every CI crossing zero |
| **WITHDRAWN** | `E0019-predictor-loss-math` (budget overrun, superseded by `E0021`); E0017's diagnosis (superseded by `E0018`); the Gemini curve (VOID). Registered in `experiments/WITHDRAWN.json` and enforced by a trap. |
| **CLOSED** | hard worst-case reservation; forced Wait units; MATH-500 as the settling benchmark (needs ~26,031 items, has 500) |
| **FUTURE HYPOTHESES** | only a genuinely new observable signal — verifier/test-execution feedback, richer uncertainty structure, intermediate tool results. Preregister, then **ceiling → predictor → controller**, never reversed. |
