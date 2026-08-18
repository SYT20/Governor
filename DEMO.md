# Governor — demo & video script

Everything here runs with **no API key and no network**. Record it as-is.

```bash
PAUSE=3 ./scripts/demo.sh        # the whole walkthrough, ~4 min
```

Recording: `asciinema rec governor.cast -c "PAUSE=3 ./scripts/demo.sh"`, or any
screen recorder at 1280×720+ with a 14–16pt monospace font.

---

## Script — 8 beats, ~5 minutes

### 0:00 — Hook (20s, slide)

> "LLMs can think longer to get better answers, but thinking costs tokens. If
> you have a budget across many tasks, **which ones deserve the extra thinking?**
> That's the allocation problem. This is a harness for studying it — and for
> being honest about whether your answer actually works."

Show the architecture diagram from `ARCHITECTURE.md` §1.

### 0:20 — Orientation (30s)

```bash
python scripts/resume_project.py
```

> "Any session starts here. It reads the repository — not notes, not memory —
> and prints the commit, the experiments, the backends, and what is frozen.
> Notice the last line: **it tells you not to trust it, and to re-verify.**"

### 0:50 — Health (30s)

```bash
python scripts/project_health.py
```

> "Two separate lines, deliberately. `ENGINEERING_HEALTH: GREEN` means the
> machinery works. `SCIENTIFIC_STATUS` is its own axis. **Green engineering
> never means the hypothesis was supported** — conflating those is the single
> most common way a research codebase lies to you."

### 1:20 — Provenance (40s)

```bash
make verify
```

> "Twenty-six experiments, each re-verified from disk: raw-file hash, row count,
> a per-run nonce, the config hash, and whether the verdict is consistent with
> its trap results. If someone edits a raw file months later, this fails."

> "It also enforces withdrawal. One experiment on disk still reads PASS — a real
> result that was later withdrawn because the policy overspent its budget by
> 15%. The row stays, because an append-only record you can edit isn't a record.
> A **trap** stops it being cited as evidence."

### 2:00 — End to end (40s)

```bash
python scripts/e2e_smoke.py
```

> "The whole stack on two different task families: calibration, seven policies,
> the executor, Ares, budget accounting, trap checks, the ledger, and the MCP
> harness. Three seconds, no API key. **Charge equals measured use, never the
> cap** — that check exists because a nominal-cost shortcut once produced a
> result we had to withdraw."

### 2:40 — Plugin (60s) ← *the part most people care about*

```bash
./scripts/mcp_smoke.sh
```

> "It's an MCP server — twelve tools over JSON-RPC stdio, no dependencies.
> Drop `configs/claude_mcp.json` into Claude Code and the Governor becomes a
> tool your agent can call."

Then the live episode:

```
t=0  H   q=+0.025 cost=0.96405 k=1 -> 330 tok, 1782 left (reserve budget for a better slot)
t=1  M2  q=+1.007 cost=0.91802 k=1 -> 585 tok, 1197 left (gain >= opportunity cost)
t=2  H   q=+1.008 cost=None    k=0 -> 330 tok,  867 left (infeasible)
t=3  M2  q=+0.922 cost=0.0     k=1 -> 585 tok,  282 left (gain >= opportunity cost)
```

> "Every decision is explained. At t=0 the predicted gain is 0.025 against an
> opportunity cost of 0.96 — **not worth it, save the budget**. At t=1 the gain
> is 1.007, clears the cost, spend. At t=2 nothing is affordable. At t=3 the
> opportunity cost has fallen to zero because there's nothing left to save for.
> That threshold falling as items run out is the whole idea, and no fixed
> schedule reproduces it."

Mention: `governor_start`, `governor_next`, `ares_execute`, `graft_get_state`,
`budget_status`, `m2_reason`, `experiment_run`, `experiment_compare`.

> "`graft_update_state` writes to a scratch slot the allocator **cannot read**.
> A writable memory that fed the controller would let a stray hint fake a
> result — a test proves a written hint can't change a decision."

### 3:40 — The traps (45s)

```bash
grep -E '^\| [0-9]+ \|' TRAPS.md
```

> "Fifteen executable checks. Each exists because of a specific failure that had
> **already printed a plausible result**. A red trap forces verdict=BLOCKED in
> the ledger — the caller doesn't get to overrule it."

> "On the final experiment the ledger recorded BLOCKED after I passed it
> INCONCLUSIVE, because a budget check went red. **The system refused my own
> verdict.** That's the property worth having."

### 4:25 — Honest ending (35s)

> "So does it work? The engineering does. The science doesn't — yet."

| axis | ceiling | Governor |
|---|---|---|
| MATH tokens | +0.164 | +0.0121 [−0.0396, +0.0510] |
| GPQA tokens | +0.232 | +0.0000 |
| LiveCodeBench samples | +0.055 | −0.0028 [−0.0066, +0.0000] |
| LCB + probe | +0.057 | +0.0046 [−0.0047, +0.0171] |

> "A perfect allocator has real headroom every time. Our controller never
> recovers it. The clearest evidence: on one run it made **45 different
> allocation decisions and not one changed an outcome** — it was reallocating
> among problems where reallocation can't matter."

> "That's a finding about **observability**, not about allocation. And the
> reason you can believe it is that the same harness killed two of our own
> positive results along the way."

---

## Using it as a plugin

```jsonc
// ~/.claude/mcp.json  (or configs/claude_mcp.json in this repo)
{
  "mcpServers": {
    "governor": {
      "command": "python",
      "args": ["-m", "governor.mcp.server"],
      "cwd": "/path/to/Atlan Proj",
      "env": { "PYTHONPATH": "." }
    }
  }
}
```

| tool | what it does |
|---|---|
| `governor_start` | open an episode, fit the predictor on calibration only |
| `governor_next` | the decision, with predicted gain, opportunity cost, and reason |
| `ares_execute` | execute one action; budget checked *before* the call |
| `governor_status` / `budget_status` | progress, spend, utilisation, affordable upgrades |
| `graft_get_state` | observable state — correctness deliberately absent |
| `graft_update_state` | scratch slot the allocator cannot read |
| `m2_reason` | call the deep arm at an explicit budget |
| `experiment_run` / `experiment_compare` / `experiment_index` | load and re-verify recorded results |
| `gate_status` | whether the ceiling gate has passed |

Every invocation is logged with tool, args, latency, git commit and error —
including calls that raised.

## Features worth naming

- **One execution path.** MCP, tests and experiments all go through the same
  `run_episode`; tests assert byte-identical traces.
- **Four engines, one contract.** MathM2, Nemotron, Gemini, local Qwen via MLX.
  The Governor cannot tell which answered; enforced against the import graph.
- **Measured cost, never nominal.** Exact tokenizer counts; `len/4` estimates
  are a red trap.
- **Ceiling before controller.** A closed-form law screens an environment in
  microseconds. `require_gate_passed()` refuses to let a controller run until a
  held-out ceiling gate has passed.
- **Provenance as a refusal.** The ledger won't finalize without commit, seeds,
  split, metric definition, raw rows and a clean tree.
