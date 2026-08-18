# Governor — Architecture

A reasoning-aware, budget-controlled compute-allocation layer for LLM agents.

The Governor decides **where scarce reasoning compute is worth spending**. It
does not reason, and it does not execute. Those are separate layers behind
frozen interfaces, which is what lets four different engines sit behind the same
contract without the controller knowing which one answered.

---

## 1. System architecture

```mermaid
graph TB
    subgraph HOST["Host"]
        CC["Claude Code / LLM host"]
        MCP["MCP server<br/><i>12 tools, JSON-RPC stdio</i>"]
    end

    subgraph CONTROL["Control plane — decides, never executes"]
        GOV["<b>GOVERNOR</b><br/>value predictor q̂(state)<br/>opportunity-cost rule"]
        GRAFT["<b>GRAFT</b><br/>observable state<br/><i>no correctness, no future</i>"]
    end

    subgraph EXEC["Execution plane"]
        ARES["<b>ARES</b><br/>execute(action, state, budget)<br/><i>budget checked BEFORE the call</i>"]
        EXECU["canonical executor<br/><i>run_episode — frozen</i>"]
    end

    subgraph ENGINES["M2 engines — frozen contract"]
        M2I["M2(state, reasoning_budget)<br/>→ M2Result"]
        MATH["MathM2<br/><i>regression anchor</i>"]
        NEM["Nemotron<br/><i>OpenRouter</i>"]
        GEM["Gemini"]
        QWEN["Qwen3-1.7B<br/><i>local, MLX</i>"]
    end

    subgraph GOV_HARNESS["Governance — blocks bad science"]
        LEDGER["experiment ledger<br/><i>refuses unreproducible results</i>"]
        TRAPS["15 executable traps<br/><i>red ⇒ verdict BLOCKED</i>"]
    end

    CC --> MCP
    MCP --> GOV
    GRAFT -- "observable state" --> GOV
    GOV -- "allocation decision" --> ARES
    ARES --> EXECU
    EXECU --> M2I
    M2I --- MATH & NEM & GEM & QWEN
    M2I -- "result + ACTUAL tokens" --> ARES
    ARES -- "observation, utility, cost" --> GRAFT
    GRAFT -. "next decision" .-> GOV

    ARES -.-> LEDGER
    GOV -.-> TRAPS
    TRAPS -. "red blocks PASS" .-> LEDGER

    classDef ctrl fill:#1e3a5f,stroke:#4a90d9,color:#fff
    classDef exec fill:#1f4d3a,stroke:#4caf50,color:#fff
    classDef eng  fill:#4a3520,stroke:#d99a4a,color:#fff
    classDef gov  fill:#4a1f2e,stroke:#d9536f,color:#fff
    class GOV,GRAFT ctrl
    class ARES,EXECU exec
    class M2I,MATH,NEM,GEM,QWEN eng
    class LEDGER,TRAPS gov
```

**The one rule that holds the design together:** there is a *single* execution
path. `tests/test_ares.py` asserts that `AresLoop` reproduces the frozen
`run_episode` byte-for-byte on two environments and two task families, and
`tests/test_mcp.py` asserts the MCP harness reproduces it too. A plugin with its
own control loop would drift from the validated one while both kept working.

---

## 2. Decision loop — one episode

```mermaid
sequenceDiagram
    participant H as Host
    participant G as Governor
    participant Gr as Graft
    participant A as Ares
    participant M as M2 engine
    participant L as Ledger

    H->>G: governor_start(family)
    G->>G: fit q̂ on CALIBRATION items only
    G->>L: open experiment (config, commit, seeds, split)

    loop each item in the episode
        Gr->>G: observable state<br/>text features, history, budget left
        G->>G: q̂(state) vs opportunity cost λ·ĉ
        alt affordable and worth it
            G->>A: execute(DEEP, state, budget_left)
        else reserve the budget
            G->>A: execute(CHEAP, state, budget_left)
        end
        A->>A: budget check BEFORE the call
        alt does not fit
            A-->>G: refused — state NOT advanced
        else fits
            A->>M: M2(state, reasoning_budget)
            M-->>A: result + ACTUAL tokens
            A->>A: charge measured cost, never nominal
            A-->>Gr: observation, utility, consumed
        end
        Gr->>Gr: update state<br/>(mode, tokens, finish_reason — NOT correctness)
    end

    G->>L: raw rows, metrics, trap results
    L->>L: verify commit, nonce, hashes, clean tree
    alt any trap red
        L-->>H: verdict = BLOCKED
    else all green
        L-->>H: verdict as recorded
    end
```

---

## 3. Experiment workflow — ceiling before controller

The order is frozen. Reversing it is how this project burned a day building a
controller for an environment with no headroom.

```mermaid
flowchart TD
    START([new task family]) --> S1{"<b>S1 headroom</b><br/>ceiling n,k,p ≥ 0.12<br/><i>closed form, microseconds</i>"}
    S1 -- fail --> REJECT1[reject the family<br/>record why]
    S1 -- pass --> S2{"<b>S2 decidability</b><br/>actual / cap ≥ 0.70"}
    S2 -- fail --> REJECT2[reject: reservation<br/>swamps preference]
    S2 -- pass --> S3{"<b>S3 stability</b><br/>k drift ≤ 0.30<br/>ceiling loss ≤ 20%"}
    S3 -- fail --> REJECT3[reject: operating point<br/>drifts between splits]
    S3 -- pass --> GATE{"<b>CEILING GATE</b><br/>oracle − greedy > 0.02<br/>CI lower bound clears it<br/><i>HELD-OUT items only</i>"}

    GATE -- fail --> CLOSED[family closed<br/>kept as negative control]
    GATE -- pass --> LOCK[["require_gate_passed()<br/>unlocks downstream"]]

    LOCK --> PRED[fit predictor<br/>CALIBRATION only<br/>AUC + Brier + ECE]
    PRED --> CTRL[build controller<br/>Lagrangian / DP]
    CTRL --> EVAL{"held-out evaluation<br/>at MATCHED REALISED COST"}
    EVAL --> TRAPCHK{15 traps}
    TRAPCHK -- "any red" --> BLOCKED[verdict BLOCKED<br/><i>caller cannot override</i>]
    TRAPCHK -- "all green" --> VERDICT[PASS / FAIL / INCONCLUSIVE]

    VERDICT --> LEDGER[(ledger:<br/>config, raw.jsonl,<br/>metrics, commit, README)]

    classDef gate fill:#4a1f2e,stroke:#d9536f,color:#fff
    classDef ok fill:#1f4d3a,stroke:#4caf50,color:#fff
    classDef bad fill:#3a3a3a,stroke:#888,color:#ddd
    class S1,S2,S3,GATE,TRAPCHK gate
    class LOCK,PRED,CTRL,VERDICT,LEDGER ok
    class REJECT1,REJECT2,REJECT3,CLOSED,BLOCKED bad
```

---

## 4. Resource contract

Three were tested on external data. Two were eliminated on measured grounds.

```mermaid
flowchart LR
    subgraph A["hard worst-case reservation ❌"]
        A1["reserve cap(mode)<br/>for every call"] --> A2["engine uses 28% of it"] --> A3["feasibility decides,<br/>not preference"]
    end
    subgraph B["forced consumption ❌"]
        B1["force N extra<br/>reasoning rounds"] --> B2["unit fully consumed ✓"] --> B3["but buys nothing:<br/>ceiling +0.009"]
    end
    subgraph C["soft expected budget ✓"]
        C1["E[Σ actual tokens] ≤ B"] --> C2["+ hard runtime cap"] --> C3["binds AND has headroom"]
    end
```

**Adopted:** `E[Σ actual tokens] ≤ B`, with a hard per-episode runtime cap and
costs charged from an exact tokenizer. `budget_adherence` (trap 14) enforces it:
a policy may under-spend, may not exceed its budget, and the baseline may never
be given fewer tokens than the policy used.

---

## 5. Frozen interfaces

| layer | contract | frozen by |
|---|---|---|
| Executor | `run_episode(env, policy, ep, budget) -> Trace` | `tests/test_reference_frozen.py` at 1e-12 |
| Ares | `execute(action, state, budget) -> ExecResult(observation, utility, consumed, state)` | trace-identical to the executor, two environments |
| M2 | `M2(state, reasoning_budget) -> M2Result(result, reasoning_tokens, total_tokens, latency_s, cost_units, ok, error)` | four backends, Governor unchanged |
| Environment | `reset / observe / step / utility / modes / cap / feasible` | `observe` carries no correctness and no future |
| Family | `features, feature_names, grade, system_prompt, heuristic_features` | swapping it is the *only* change needed for a new task family |
| Ledger | `ExperimentRun.finalize(...)` refuses rather than records | 13 provenance tests |

---

## 6. Status

| axis | state |
|---|---|
| **ENGINEERING** | GREEN — 257 tests, 26/26 experiments verify, smoke passes |
| **SCIENCE** | real-LLM Governor advantage **NOT VERIFIED** — 4 experiments, 3 axes, every CI crossing zero |

These are different axes. GREEN engineering does not mean the hypothesis held.
See `FINAL-CLAIMS.md` for the full VERIFIED / NOT VERIFIED / WITHDRAWN matrix.
