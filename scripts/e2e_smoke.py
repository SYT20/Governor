#!/usr/bin/env python3
"""End-to-end smoke test. No API key, no quota, no network. Deterministic.

Exercises the whole stack in one pass:

    task family -> environment -> calibration -> Governor -> Ares -> executor
    -> budget accounting -> trap checks -> ledger -> MCP tools

and asserts the pieces agree with each other, which is the property that keeps
a plugin from drifting away from the thing that was validated. Runs on both task
families so nothing is quietly specialised to one.

Exit code 0 means the system is wired correctly. It says nothing about whether
any scientific claim is true -- that is what the experiments are for.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from governor.ares.executor import AresLoop  # noqa: E402
from governor.gate.executor import run_episode  # noqa: E402
from governor.gate.m2_interface import MathM2  # noqa: E402
from governor.harness.ledger import git_commit, index  # noqa: E402
from governor.harness.traps import render, run_trap_checks  # noqa: E402
from governor.mcp import sessions as S  # noqa: E402
from governor.mcp.server import call_tool, handle, Recorder, tool_schemas  # noqa: E402
from governor.phase4.collect import CallRecord, ResponseCache  # noqa: E402
from governor.phase4.config import PROMPT_CAP  # noqa: E402
from governor.phase4.env import DEEP, P4Env, make_episodes  # noqa: E402
from governor.phase4.evaluate import constant, execute, token_evidence  # noqa: E402
from governor.phase4.family import ARITHMETIC, PUZZLES  # noqa: E402
from governor.phase4.pipeline import calibrate, evaluate_heldout, summarise  # noqa: E402
from governor.phase4.policies import all_cheap, greedy  # noqa: E402
from governor.phase4 import puzzles as PZ, tasks as AR  # noqa: E402

OK, FAIL = "  ok  ", " FAIL "
LOW, HIGH, N = 300, 700, 6
# Room for about two of six upgrades: scarce enough that WHICH items get
# them matters, which is the whole point of the check below.
BUDGET = N * (PROMPT_CAP + LOW) + 2.0 * (HIGH - LOW)
_fails: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  [{OK if cond else FAIL}] {name}{('  ' + detail) if detail else ''}")
    if not cond:
        _fails.append(name)


def _cache(tmp: Path, family, pool, reply):
    """A world with GENUINE allocation value.

    The first version keyed difficulty off an arbitrary feature index and
    produced a world where every item was alike, so the Governor collapsed into
    greedy and the trap checks went red -- correctly. A smoke test asserting
    "traps green" has to supply a world in which an allocator can actually do
    something, or it is asserting that the traps are broken.

    Difficulty is driven by the family's OWN named difficulty feature, split at
    its median, so roughly half the items benefit while the budget affords far
    fewer -- the S1 condition, in miniature.
    """
    c = ResponseCache(tmp, model="smoke", system_prompt=family.system_prompt)
    key = family.difficulty_feature
    vals = np.array([family.features(i.prompt)[key] for i in pool], float)
    cut = float(np.median(vals))
    for it in pool:
        hard = family.features(it.prompt)[key] >= cut
        for mt in (LOW, HIGH):
            good = (not hard) or mt == HIGH
            used = int(mt * (0.9 if mt == LOW else 0.7))
            c.put(it, mt, CallRecord(it.item_id, mt, reply(it, good), "stop",
                                     60, used, used - 10, 60 + used, 0.01), 1)
    return c


def main() -> int:
    import tempfile
    tmp = Path(tempfile.mkdtemp())
    t0 = time.time()
    print("=" * 74)
    print(f"END-TO-END SMOKE TEST   commit {git_commit()[:8]}")
    print("=" * 74)

    print("\nM2 contract")
    r = MathM2()({}, 1.0)
    check("MathM2 honours the frozen contract", r.ok and r.cost_units == 1.0)
    check("under-budget call does not charge", not MathM2()({}, 0.5).ok)

    for fam, mod, reply in (
            (ARITHMETIC, AR, lambda it, g: str(it.answer) if g else "0"),
            (PUZZLES, PZ, lambda it, g: "<think>x</think>" + ", ".join(
                f"{k}={v if g else 1}" for k, v in it.answer.items()))):
        print(f"\nfamily: {fam.name}")
        pool = mod.make_pool(seed=11, n=96)
        cache = _cache(tmp / f"{fam.name}.sqlite", fam, pool, reply)
        cal_items, test_items = pool[:48], pool[48:]
        cal = P4Env(cache, make_episodes(cal_items, 8, 1, n_items=N), LOW, HIGH,
                    BUDGET, PROMPT_CAP, family=fam)
        test = P4Env(cache, make_episodes(test_items, 8, 2, n_items=N), LOW, HIGH,
                     BUDGET, PROMPT_CAP, family=fam)
        C, T = list(range(8)), list(range(8))

        c = calibrate(cal, cal_items, C)
        check("calibration fits a predictor", c.predictor.model is not None,
              f"kind={c.predictor_kind} cv_R2={c.report.cv_r2:+.3f}")
        R, trace = evaluate_heldout(test, c, T)
        check("all seven policies executed", len(R) == 7)
        check("no policy overspends",
              all(r.spent.max() <= test.budget + 1e-9 for r in R.values()))
        check("every item answered",
              all(all(len(m) == N for m in r.modes) for r in R.values()))
        check("oracle bounds every policy",
              all(r.mean <= R["oracle"].mean + 1e-9
                  for k, r in R.items() if k != "oracle"))
        te = token_evidence(R["GOVERNOR"], test)
        check("charge equals measured use, never the cap",
              np.allclose(te["actual_used"], te["charged"])
              and all(a <= q for a, q in zip(te["actual_used"], te["requested"])))

        loop = AresLoop(test)
        same = all(loop.run(all_cheap(test), e, test.budget).actions
                   == run_episode(test, all_cheap(test), e, test.budget).modes
                   and loop.run(greedy(test), e, test.budget).costs
                   == run_episode(test, greedy(test), e, test.budget).costs
                   for e in T)
        check("Ares is trace-identical to the frozen executor", same)

        s = summarise(R, c, test, trace, commit="smoke", froze_commit="prereg",
                      selection_item_ids=[i.item_id for i in cal_items],
                      evaluation_item_ids=[i.item_id for i in test_items])
        check("Governor is not a schedule in disguise",
              len({m for m in R["GOVERNOR"].modes}) > 1,
              f"{len({m for m in R['GOVERNOR'].modes})} distinct patterns")
        check("trap checks green", s["red"] == [], str(s["red"]))
        if s["red"]:
            print(render(s["traps"]))

    print("\nledger")
    rows = index()
    check("experiments still verify",
          all(r["verifies"] for r in rows if r["verdict"] != "UNFINALIZED"),
          f"{sum(r['verifies'] for r in rows)}/{len(rows)}")

    print("\nMCP harness")
    S.reset_sessions()
    rec = Recorder(root=tmp)
    init = handle({"jsonrpc": "2.0", "id": 1, "method": "initialize"}, rec)
    check("initialize", init["result"]["serverInfo"]["name"] == "governor")
    check("tools advertised", len(tool_schemas()) >= 10, f"{len(tool_schemas())}")
    sid = call_tool("governor_start", {"family": "synthetic"}, rec)["session_id"]
    steps = 0
    while True:
        d = call_tool("governor_next", {"session_id": sid}, rec)
        if d.get("done"):
            break
        e = call_tool("ares_execute",
                      {"session_id": sid, "action": d["action"]}, rec)
        steps += 1
        if e["done"]:
            break
    st = call_tool("governor_status", {"session_id": sid}, rec)
    check("MCP episode completes", st["done"] and steps == st["n_decisions"])
    check("MCP respects the budget", st["spent"] <= st["budget"] + 1e-9)
    g = call_tool("graft_get_state", {"session_id": sid}, rec)
    check("graft state hides correctness", "correct" not in str(g["outcomes_observable"]))
    check("every tool call recorded", rec.path.exists()
          and len(rec.path.read_text().splitlines()) >= steps * 2)
    gate = call_tool("gate_status", {}, rec)
    check("gate status reported", "verdict" in gate, gate.get("verdict", ""))

    print("\n" + "=" * 74)
    if _fails:
        print(f"SMOKE TEST FAILED: {len(_fails)} check(s) — {_fails}")
        return 1
    print(f"SMOKE TEST PASSED — whole stack wired, both families, "
          f"{time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
