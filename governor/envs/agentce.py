"""AgentCE-Bench adapter — environment 2, with Governor's accountant in the loop.

Verified properties (by execution, 2026-08-15, not by citation):
  * repo github.com/uservan/AgentCE_Bench (MIT), arXiv 2604.06111. The name is
    AgentCE-Bench; "ACE-Bench" is a different benchmark.
  * 324 instances = 54 x 6 domains, 31.2 MB for the 5x7 set.
  * H (hidden slots)  in {1,5,7,11,15,21}   -- task horizon
  * B (branch budget) in {0,2,4,8,10,15,19,21,25} -- DECOY GENERATION only.
    Confirmed from the instance fields `branch_budget_allocations` and
    `decoy_generation_final_stage`: B controls how many misleading candidates
    exist. It is NOT a resource limit. So H, B and our resource envelope R are
    genuinely orthogonal axes.
  * Runs with NO LLM. Task.call_tool() dispatches straight to the tool handler.
  * Oracle check: writing truth_solution into the hidden slots evaluates to
    {"score": true}, so scoring is wired correctly.

Three distinct budget notions exist here and must not be conflated:
  1. B, the decoy budget          -- difficulty, set by the dataset
  2. per-slot query budget and global-check budget -- the benchmark's own
     built-in limits on how much the agent may interrogate the environment
  3. R, our resource envelope     -- tokens / cost / wall-clock / tool calls,
     imposed and enforced by Governor's Accountant
Only (3) is ours. (2) is a property of the benchmark that our policy must
respect, and it is itself a resource-allocation problem, which is why this
environment suits the project.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from governor.accounting.meter import Accountant, Envelope

AGENTCE_ROOT = Path("/tmp/AgentCE_Bench")
DOMAINS = ("course", "meal", "pc_build", "shopping", "travel", "workforce")

# Cost model for the R axis. AgentCE resolves tools from static JSON, so there is
# no natural monetary cost -- we impose one, scaled by how much information each
# call yields. These are OUR parameters and are declared here rather than buried,
# because every downstream budget result depends on them.
TOOL_COST = {
    "check_slot_constraints": {"tokens": 320.0, "cost": 0.0016, "wall_s": 0.25, "tool_calls": 1.0},
    "query_candidate":        {"tokens": 900.0, "cost": 0.0045, "wall_s": 0.60, "tool_calls": 1.0},
    "get_item_attributes":    {"tokens": 480.0, "cost": 0.0024, "wall_s": 0.35, "tool_calls": 1.0},
    "get_item_info":          {"tokens": 260.0, "cost": 0.0013, "wall_s": 0.20, "tool_calls": 1.0},
    "check_global":           {"tokens": 700.0, "cost": 0.0035, "wall_s": 0.50, "tool_calls": 1.0},
    "grid_state":             {"tokens": 200.0, "cost": 0.0010, "wall_s": 0.15, "tool_calls": 1.0},
    "set_slot":               {"tokens": 120.0, "cost": 0.0006, "wall_s": 0.10, "tool_calls": 1.0},
    "done":                   {"tokens": 60.0,  "cost": 0.0003, "wall_s": 0.05, "tool_calls": 0.0},
}
_GENERIC = {"set_slot", "get_current_grid_state", "get_slot_id", "done",
            "get_hidden_slot_query_budget", "get_global_check_budget"}


class ToolCallError(RuntimeError):
    """A tool reported failure. Never silently degrade this into an empty result."""


def _ensure_path() -> None:
    if str(AGENTCE_ROOT) not in sys.path:
        sys.path.insert(0, str(AGENTCE_ROOT))


def available() -> bool:
    """True when the benchmark is present. Keeps the rest of the suite runnable."""
    return (AGENTCE_ROOT / "env" / "agent" / "task.py").exists()


def load_instances(domain: str, grid: str = "5x7") -> list:
    _ensure_path()
    from env.load_datasets.loader import load_dataset_objects_from_file
    d = AGENTCE_ROOT / "data" / grid
    hits = sorted(d.glob(f"{domain}_*.json"))
    if not hits:
        raise FileNotFoundError(f"no dataset for domain {domain!r} in {d}")
    return load_dataset_objects_from_file(str(hits[0]))


def resolve_field(rule_name: str, item_pool: dict) -> str | None:
    """Map a slot-constraint rule name onto the actual item attribute name.

    Naive prefix-stripping is WRONG and cost me a silent 0% on an entire domain:
    travel's rule `max_crowd` refers to the item field `crowd_level`, so stripping
    to `crowd` produced a failed query on every single travel instance while
    course and meal happened to work. The failure was invisible because a failed
    query simply returns no matches, which looks like an empty candidate pool.

    Resolve against the real item schema instead of guessing: exact match first,
    then unique prefix match, then unique substring match. Return None rather than
    guessing when ambiguous.
    """
    base = rule_name.replace("max_", "", 1).replace("min_", "", 1)
    sample = next(iter(item_pool.values()), None) if isinstance(item_pool, dict) else None
    if not isinstance(sample, dict):
        return base
    fields = [f for f in sample if not f.endswith("_id") and f != "name"]
    if base in fields:
        return base
    pref = [f for f in fields if f.startswith(base)]
    if len(pref) == 1:
        return pref[0]
    sub = [f for f in fields if base in f]
    if len(sub) == 1:
        return sub[0]
    return None


def cost_key(tool_name: str) -> str:
    if tool_name in _GENERIC:
        return {"set_slot": "set_slot", "done": "done"}.get(tool_name, "grid_state")
    for frag, key in (("slot_constraints", "check_slot_constraints"),
                      ("global_constraints", "check_global"),
                      ("candidate_from_attribute", "query_candidate"),
                      ("item_attributes", "get_item_attributes"),
                      ("item_info", "get_item_info")):
        if frag in tool_name:
            return key
    return "grid_state"


@dataclass(slots=True)
class Step:
    """One metered tool call. The audit trail for a decision."""
    seq: int
    tool: str
    args: dict
    ok: bool
    charged: dict
    truncated: bool


@dataclass(slots=True)
class AgentCEEpisode:
    """A single AgentCE task run under Governor's resource envelope.

    The Accountant is the same one validated in SynthBug: it meters every tool
    call and refuses to dispatch once the envelope is exhausted, so BVR = 0 holds
    here by the same structural argument.
    """

    instance: Any
    envelope: Envelope
    tool_failure_rate: float = 0.0
    seed: int = 0
    max_steps: int = 200
    strict: bool = True          # raise on tool failure; see call()
    task: Any = field(init=False, default=None)
    acc: Accountant = field(init=False, default=None)
    steps: list[Step] = field(default_factory=list)
    exhausted: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        _ensure_path()
        from env.agent.task import Task
        self.task = Task(dataset_object=self.instance, max_steps=self.max_steps,
                         tool_failure_rate=self.tool_failure_rate, seed=self.seed)
        self.acc = Accountant(envelope=self.envelope)

    # -- properties the policy may read ------------------------------------
    @property
    def domain(self) -> str:
        return self.instance.domain

    @property
    def hidden_slots(self) -> list[tuple[int, int]]:
        return list(self.task.hidden_slot_path)

    @property
    def horizon(self) -> int:
        return len(self.task.hidden_slot_path)

    @property
    def decoy_budget(self):
        return self.task.branch_budget

    def slot_query_budget(self, row: int, col: int):
        return self.task.get_remaining_hidden_slot_queries(row, col)

    def global_check_budget(self):
        return self.task.get_remaining_global_checks()

    # -- metered execution --------------------------------------------------
    def call(self, tool: str, **args: Any) -> Any:
        """Execute a tool, charging the envelope. Refuses once exhausted.

        This is the Ares-like execution boundary: the policy names an action, the
        executor runs it and reports what it actually cost, and the accountant
        enforces the envelope. Nothing here estimates.
        """
        price = TOOL_COST[cost_key(tool)]
        rem = self.acc.remaining()
        if any(rem[d] < price[d] for d in price):
            self.exhausted = True
            return {"error": "BUDGET_EXHAUSTED"}
        t0 = time.monotonic()
        try:
            out = self.task.call_tool(tool, args)
            ok = True
        except Exception as exc:                      # tool misuse is a real outcome
            out, ok = {"error": f"{type(exc).__name__}: {exc}"}, False

        # ENFORCED INVARIANT, not documentation. A malformed query returns
        # {"status": "failed", "data": {}}, which is indistinguishable from a
        # legitimate empty result. That ambiguity has now produced two wrong
        # conclusions in this project:
        #   1. `max_crowd` -> field `crowd` (real field: crowd_level) made every
        #      travel query fail; the domain scored 0% and I nearly reported it as
        #      a property of the benchmark.
        #   2. get_item_info on hidden-slot ids fails by design ("You may only
        #      inspect ids from non-hidden slots"); the silent failure produced an
        #      empty findings table that I read as "no information channel exists",
        #      which was the opposite of the truth.
        # Failing loudly by default is the only thing that reliably stops this.
        if isinstance(out, dict) and out.get("status") == "failed" and self.strict:
            raise ToolCallError(
                f"{tool}({args}) -> {out.get('messages')!r}. "
                "Pass strict=False only where a failure is the expected outcome."
            )
        charged = dict(price)
        charged["wall_s"] = min(max(time.monotonic() - t0, price["wall_s"]),
                                max(rem["wall_s"], 0.0))
        self.acc.charge(tool, **charged)
        self.steps.append(Step(len(self.steps), tool, dict(args), ok, charged, False))
        return out

    # -- scoring -------------------------------------------------------------
    def score(self) -> bool:
        res = self.task.eval()
        return bool(res.get("score")) if isinstance(res, dict) else bool(res)

    def summary(self) -> dict:
        return {"domain": self.domain, "H": self.horizon, "B": self.decoy_budget,
                "steps": len(self.steps), "success": self.score(),
                "exhausted": self.exhausted, **self.acc.summary()}
