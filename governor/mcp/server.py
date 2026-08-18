"""MCP stdio server. JSON-RPC 2.0 over stdin/stdout, no third-party deps.

Every invocation is recorded with experiment id, task, model, budget, allocated
budget, actual cost, decision, state transition, result, latency and git commit
-- appended to `experiments/MCP-<date>/raw.jsonl` through the same ledger the
experiments use. A tool call that leaves no trace is the one that will later be
quoted from memory.

The tools delegate to `governor.mcp.sessions`, which delegates to the same
Governor, ActionExecutor and executor the tests exercise. There is no second
implementation anywhere in this file.
"""
from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable

from governor.harness.ledger import EXPERIMENTS, git_commit
from governor.mcp import sessions as S

PROTOCOL = "2024-11-05"
SERVER = {"name": "governor", "version": "1.0.0"}


def _experiment_run(exp_id: str) -> dict:
    """Report a recorded experiment. Does NOT execute a new scientific run --
    an experiment must be launched deliberately, with a commit, not by a tool
    call that could be repeated until it says something nice."""
    from governor.harness.ledger import load_experiment, verify_experiment
    res = load_experiment(exp_id)
    ok, problems = verify_experiment(exp_id)
    return {"exp_id": exp_id, "verdict": res["verdict"],
            "summary": res["summary"], "commit": res["git_commit"],
            "raw_rows": res["raw_rows"], "verifies": ok, "problems": problems,
            "red_traps": res["red_traps"]}


def _experiment_compare(exp_a: str, exp_b: str) -> dict:
    a, b = _experiment_run(exp_a), _experiment_run(exp_b)
    return {"a": a, "b": b,
            "same_commit": a["commit"] == b["commit"],
            "both_verify": a["verifies"] and b["verifies"],
            "verdicts": {exp_a: a["verdict"], exp_b: b["verdict"]}}


def _index() -> dict:
    from governor.harness.ledger import index
    return {"experiments": index()}


def _gate() -> dict:
    from governor.phase4.gatekeeper import GATE_EXP, gate_status
    return {"gate_experiment": GATE_EXP, **gate_status()}


TOOLS: dict[str, tuple[Callable, str, dict]] = {
    "governor_start": (S.governor_start,
        "Open an episode and fit the observable value predictor on the "
        "calibration split. Phase 4R is refused until its ceiling gate passes.",
        {"family": {"type": "string", "enum": sorted(S.FAMILIES)},
         "episode": {"type": "integer"}}),
    "governor_next": (S.governor_next,
        "The Governor's decision for the current item: predicted gain vs the "
        "opportunity cost of the budget it would consume. Decides, does not "
        "execute.",
        {"session_id": {"type": "string"}}),
    "ares_execute": (S.ares_execute,
        "Execute one action through ActionExecutor. The only way a session advances; "
        "budget is checked before the call and charged at measured cost.",
        {"session_id": {"type": "string"}, "action": {"type": "string"}}),
    "governor_status": (S.governor_status, "Episode progress and decision log.",
        {"session_id": {"type": "string"}}),
    "budget_status": (S.budget_status,
        "Budget, spend, utilisation and how many upgrades remain affordable.",
        {"session_id": {"type": "string"}}),
    "m2_reason": (S.m2_reason,
        "Call the deep arm on the current item at an explicit token budget, "
        "through the same response cache the environment reads.",
        {"session_id": {"type": "string"}, "budget": {"type": "integer"}}),
    "graft_get_state": (S.graft_get_state,
        "Cognitive state: observations, reasoning history, actions, observable "
        "outcomes, uncertainty, remaining budget. Correctness is absent by "
        "design.", {"session_id": {"type": "string"}}),
    "graft_update_state": (S.graft_update_state,
        "Write to an isolated scratch slot the allocator never reads.",
        {"session_id": {"type": "string"}, "key": {"type": "string"},
         "value": {}}),
    "experiment_run": (_experiment_run,
        "Load a recorded experiment and re-verify it from disk.",
        {"exp_id": {"type": "string"}}),
    "experiment_compare": (_experiment_compare, "Compare two recorded experiments.",
        {"exp_a": {"type": "string"}, "exp_b": {"type": "string"}}),
    "experiment_index": (_index, "Every experiment and whether it still verifies.", {}),
    "gate_status": (_gate, "Whether the Phase 4R ceiling gate has passed.", {}),
}


class Recorder:
    """Append-only provenance for tool calls."""

    def __init__(self, root: Path | None = None):
        d = (root or EXPERIMENTS) / f"MCP-{time.strftime('%Y%m%d')}"
        d.mkdir(parents=True, exist_ok=True)
        self.path = d / "raw.jsonl"
        self.commit = git_commit()

    def log(self, tool: str, args: dict, result: Any, latency: float,
            error: str = "") -> None:
        row = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "tool": tool, "args": args, "latency_s": round(latency, 4),
               "commit": self.commit, "error": error}
        if isinstance(result, dict):
            for k in ("session_id", "family", "action", "consumed_tokens",
                      "spent", "budget_left", "utility", "done",
                      "predicted_gain", "opportunity_cost", "verdict"):
                if k in result:
                    row[k] = result[k]
        try:
            with self.path.open("a") as f:
                f.write(json.dumps(row, default=str) + "\n")
        except OSError:
            pass


def call_tool(name: str, args: dict, rec: Recorder | None = None) -> dict:
    if name not in TOOLS:
        raise KeyError(f"unknown tool {name!r}")
    fn = TOOLS[name][0]
    t0 = time.time()
    try:
        out = fn(**args)
    except Exception as e:                                   # noqa: BLE001
        if rec:
            rec.log(name, args, None, time.time() - t0, error=str(e)[:300])
        raise
    if rec:
        rec.log(name, args, out, time.time() - t0)
    return out


def tool_schemas() -> list[dict]:
    return [{"name": n,
             "description": d,
             "inputSchema": {"type": "object", "properties": p,
                             "required": [k for k in p
                                          if k not in ("family", "episode",
                                                       "value")]}}
            for n, (_, d, p) in TOOLS.items()]


def handle(msg: dict, rec: Recorder) -> dict | None:
    mid, method = msg.get("id"), msg.get("method")
    try:
        if method == "initialize":
            result = {"protocolVersion": PROTOCOL, "serverInfo": SERVER,
                      "capabilities": {"tools": {}}}
        elif method == "tools/list":
            result = {"tools": tool_schemas()}
        elif method == "tools/call":
            p = msg.get("params") or {}
            out = call_tool(p.get("name", ""), p.get("arguments") or {}, rec)
            result = {"content": [{"type": "text",
                                   "text": json.dumps(out, indent=2, default=str)}]}
        elif method in ("notifications/initialized", "initialized"):
            return None
        elif method == "ping":
            result = {}
        else:
            raise KeyError(f"unknown method {method!r}")
        return {"jsonrpc": "2.0", "id": mid, "result": result}
    except Exception as e:                                   # noqa: BLE001
        return {"jsonrpc": "2.0", "id": mid,
                "error": {"code": -32000, "message": str(e)[:400],
                          "data": traceback.format_exc()[-800:]}}


def main() -> int:
    rec = Recorder()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        resp = handle(msg, rec)
        if resp is not None:
            sys.stdout.write(json.dumps(resp, default=str) + "\n")
            sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
