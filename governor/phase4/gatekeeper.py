"""The sequential protocol, enforced in code.

Nothing downstream of the ceiling gate may run until the gate has RECORDED a
pass on untouched evaluation items. Not the Governor, not a predictor fit, not
a hyperparameter search.

WHY THIS IS A MODULE AND NOT A PROMISE. Twice in this project a rule held only
because of how a script happened to be written: the evaluation split was a
`pool[:40]` slice that would have moved as collection proceeded, and the
collector was evaluation-only only because every selection item already had its
cheap response. Both were correct on the day and structurally unsound. "I will
not build the Governor early" is the same kind of guarantee, so it is a
function that raises instead.
"""
from __future__ import annotations

from governor.harness.ledger import EXPERIMENTS, load_experiment, verify_experiment

GATE_EXP = "E0006-ceiling-gate"
PASS_VERDICT = "CEILING-PASS"


class GateNotPassed(RuntimeError):
    """Raised instead of letting a controller be built on an unvalidated family."""


def gate_status(exp_id: str = GATE_EXP) -> dict:
    d = EXPERIMENTS / exp_id
    if not (d / "results.json").exists():
        return {"exists": False, "verdict": None, "verifies": False}
    res = load_experiment(exp_id)
    ok, problems = verify_experiment(exp_id)
    ho = (res.get("summary") or {}).get("held_out") or {}
    return {"exists": True, "verdict": res["verdict"], "verifies": ok,
            "problems": problems, "held_out": ho,
            "ci_lo": ho.get("ci_lo"), "n_items": ho.get("n_items")}


def require_gate_passed(exp_id: str = GATE_EXP) -> dict:
    """Call this FIRST in any Phase 4R controller script.

    Requires the gate experiment to exist, to still verify from disk, to carry
    the pass verdict, and to have reached that verdict on held-out items --
    an in-selection pass is explicitly not enough.
    """
    s = gate_status(exp_id)
    if not s["exists"]:
        raise GateNotPassed(
            f"{exp_id} has not been run. The ceiling gate must pass on "
            f"untouched evaluation items before any controller is built.")
    if not s["verifies"]:
        raise GateNotPassed(f"{exp_id} no longer verifies from disk: "
                            f"{s['problems']}")
    if s["verdict"] != PASS_VERDICT:
        raise GateNotPassed(
            f"{exp_id} verdict is {s['verdict']!r}, not {PASS_VERDICT!r}. "
            f"Nothing downstream may run. If the gate failed, the family is "
            f"rejected and kept as a negative control.")
    if not s["held_out"]:
        raise GateNotPassed(
            f"{exp_id} passed, but on no held-out items. An in-selection "
            f"ceiling is not the gate.")
    return s
