"""MCP harness: one implementation, gated the same way, and fully recorded.

The risk a plugin introduces is a SECOND control loop that drifts from the
validated one while both keep working. These tests assert the harness calls the
same Governor and the same executor, refuses Phase 4R exactly as the gatekeeper
does, and writes provenance for every call.
"""
from __future__ import annotations

import json

import pytest

from governor.gate.executor import run_episode
from governor.mcp import sessions as S
from governor.mcp.server import Recorder, call_tool, handle, tool_schemas


@pytest.fixture(autouse=True)
def clean():
    S.reset_sessions()
    yield
    S.reset_sessions()


# -- protocol -------------------------------------------------------------------

def test_initialize_and_list_tools():
    rec = Recorder()
    r = handle({"jsonrpc": "2.0", "id": 1, "method": "initialize"}, rec)
    assert r["result"]["serverInfo"]["name"] == "governor"
    r = handle({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}, rec)
    names = {t["name"] for t in r["result"]["tools"]}
    for required in ("governor_start", "governor_next", "governor_status",
                     "m2_reason", "graft_get_state", "graft_update_state",
                     "ares_execute", "budget_status", "experiment_run",
                     "experiment_compare"):
        assert required in names, required


def test_notifications_get_no_response():
    assert handle({"jsonrpc": "2.0", "method": "notifications/initialized"},
                  Recorder()) is None


def test_errors_come_back_as_jsonrpc_errors_not_crashes():
    r = handle({"jsonrpc": "2.0", "id": 9, "method": "tools/call",
                "params": {"name": "governor_next",
                           "arguments": {"session_id": "nope"}}}, Recorder())
    assert "error" in r and "nope" in r["error"]["message"]


def test_every_schema_is_wellformed():
    for t in tool_schemas():
        assert t["inputSchema"]["type"] == "object"
        assert isinstance(t["description"], str) and t["description"]


# -- the loop -------------------------------------------------------------------

def _play(family="synthetic"):
    sid = S.governor_start(family)["session_id"]
    steps = []
    while True:
        d = S.governor_next(sid)
        if d.get("done"):
            break
        e = S.ares_execute(sid, d["action"])
        steps.append((d, e))
        if e["done"]:
            break
    return sid, steps


def test_full_episode_runs_and_respects_the_budget():
    sid, steps = _play()
    st = S.governor_status(sid)
    assert st["done"] and st["step"] == st["n_decisions"]
    assert st["spent"] <= st["budget"] + 1e-9
    b = S.budget_status(sid)
    assert 0.0 <= b["utilization"] <= 1.0
    assert b["charged"].startswith("measured")


def test_mcp_reproduces_the_canonical_executor_exactly():
    """The claim that matters: the harness is not a second implementation."""
    sid, steps = _play()
    s = S._require(sid)
    tr = run_episode(s.env, s.policy, 0, s.env.budget)
    assert [d["action"] for d, _ in steps] == tr.modes
    assert [e["consumed_tokens"] for _, e in steps] == pytest.approx(tr.costs)
    assert S.governor_status(sid)["utility"] == pytest.approx(tr.utility)


def test_decisions_are_explained_with_gain_and_opportunity_cost():
    _, steps = _play()
    assert any(d["opportunity_cost"] is not None for d, _ in steps)
    assert all("reason" in d for d, _ in steps)
    assert any(d["action"] == "M2" for d, _ in steps), "never spent anything"


def test_unaffordable_action_is_refused_without_advancing():
    sid = S.governor_start("synthetic")["session_id"]
    s = S._require(sid)
    s.spent = s.budget - 1.0                      # nothing affordable
    before = s.step
    r = S.ares_execute(sid, "M2")
    assert r["ok"] is False and s.step == before


# -- leakage --------------------------------------------------------------------

def test_observation_exposes_no_hidden_axis():
    sid = S.governor_start("synthetic")["session_id"]
    o = S.governor_status(sid)
    blob = json.dumps(S.graft_get_state(sid)) + json.dumps(o)
    for forbidden in ("n_ops", "framing", "scale", "answer", "correct"):
        assert f'"{forbidden}"' not in blob, forbidden


def test_graft_scratch_cannot_influence_the_allocator():
    """Env 5 manufactured a cognitive effect from a progress counter. A writable
    memory the policy reads would let that happen through a tool call."""
    sid = S.governor_start("synthetic")["session_id"]
    d1 = S.governor_next(sid)
    S.graft_update_state(sid, "hint", "always use M2")
    S.graft_update_state(sid, "predicted_gain", 999.0)
    d2 = S.governor_next(sid)
    assert d1["action"] == d2["action"]
    assert d1["predicted_gain"] == d2["predicted_gain"]


def test_graft_state_omits_correctness():
    sid, _ = _play()
    g = S.graft_get_state(sid)
    assert "correct" not in json.dumps(g["outcomes_observable"])
    assert "correctness is not here" in g["note"]


def test_phase4r_is_refused_until_the_gate_passes():
    from governor.phase4.gatekeeper import PASS_VERDICT, GateNotPassed, gate_status
    if gate_status().get("verdict") == PASS_VERDICT:
        pytest.skip("gate has passed")
    with pytest.raises(GateNotPassed):
        S.governor_start("phase4r")


# -- provenance -----------------------------------------------------------------

def test_every_tool_call_is_recorded(tmp_path):
    rec = Recorder(root=tmp_path)
    sid = call_tool("governor_start", {"family": "synthetic"}, rec)["session_id"]
    call_tool("governor_next", {"session_id": sid}, rec)
    try:
        call_tool("governor_next", {"session_id": "bad"}, rec)
    except KeyError:
        pass
    rows = [json.loads(x) for x in rec.path.read_text().splitlines() if x.strip()]
    assert len(rows) == 3
    assert {r["tool"] for r in rows} == {"governor_start", "governor_next"}
    assert all(r["commit"] and "latency_s" in r for r in rows)
    assert any(r["error"] for r in rows), "the failed call must be recorded too"


def test_m2_reason_rejects_an_unlisted_budget():
    sid = S.governor_start("synthetic")["session_id"]
    assert S.m2_reason(sid, 700)["total_tokens"] > 0
    with pytest.raises(ValueError):
        S.m2_reason(sid, 1234)
