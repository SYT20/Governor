# Reproducing this project

Everything below runs on a laptop with **no API key**, except the two commands
explicitly marked. Python 3.12, numpy, scikit-learn. MLX only for the optional
local backend.

```bash
make test      # 222 tests: unit, integration, executor traces, ledger, traps, MCP
make smoke     # end-to-end: both task families, Ares, MCP, traps, ledger
make verify    # re-verify every recorded experiment from disk, months later
make mcp       # MCP server over real stdio JSON-RPC
```

`make verify` is the one that matters for trusting the numbers: it re-hashes
every `raw.jsonl`, re-checks row counts, nonces, config hashes, and that no
experiment claims a pass while carrying a red trap.

## Needs credentials

```bash
export Groq=...            # or OR_KEY=... for OpenRouter
make collect               # fills evaluation items as the per-day bucket refills
make gate                  # Phase 4R held-out ceiling gate
make governor              # refuses unless the gate recorded CEILING-PASS
```

Collection is resumable and idempotent — it fetches only what the cache lacks.
**Commit before a long run**: the ledger refuses to finalize if HEAD moves.

## Quota, measured

| provider | limit | consequence |
|---|---|---|
| Groq | TPM 8000, RPD 1000, **TPD 200,000 on RESERVED tokens** | TPD binds first: ~159 items/day at the Phase 4R config |
| OpenRouter free | 50 requests/day | nemotron could not produce a 160-call curve at all |

Groq also returns Cloudflare 403 to the default `Python-urllib` User-Agent,
which looks exactly like an auth failure and is not one.

## Layout

```
governor/gate/        frozen executor, M2 contract, engines (Math, LLM, Gemini, Qwen-local)
governor/phase4/      task families, environment, predictor + DP, policies, pipeline
governor/ares/        per-action execution layer, trace-identical to the executor
governor/harness/     experiment ledger, 12 trap checks
governor/mcp/         12-tool JSON-RPC harness over the same Governor
experiments/E000N/    config, results, metrics, raw.jsonl, git_commit, README
configs/              frozen selection/evaluation split, MCP server config
```
