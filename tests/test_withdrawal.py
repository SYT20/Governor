"""Withdrawn results must stay withdrawn.

The ledger is append-only, so `E0019-predictor-loss-math` still reads PASS on
disk even though its Governor overspent by 15%. Preserving that row is correct.
Citing it as evidence is not, and nothing but discipline prevented that until
these tests existed.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from governor.harness.ledger import index, withdrawn
from governor.harness.traps import run_trap_checks, withdrawn_result_promotion

ROOT = Path(__file__).resolve().parents[1]
CLAIMS = ROOT / "FINAL-CLAIMS.md"


def test_register_is_wellformed_and_points_at_replacements():
    w = withdrawn()
    assert w, "withdrawal register is empty"
    ids = {r["exp_id"] for r in index()}
    for exp, rec in w.items():
        assert exp in ids, f"{exp} withdrawn but not in the ledger"
        for field in ("reason", "superseded_by", "recorded_verdict"):
            assert rec.get(field), f"{exp} missing {field}"
        assert rec["superseded_by"] in ids, (
            f"{exp} superseded by {rec['superseded_by']}, which does not exist")


def test_the_known_bad_pass_is_registered():
    """The specific defect: a PASS obtained by overspending the budget."""
    w = withdrawn()
    assert "E0019-predictor-loss-math" in w
    rec = w["E0019-predictor-loss-math"]
    assert rec["recorded_verdict"] == "PASS"
    assert rec["superseded_by"] == "E0021-enforced-math"
    assert "973" in rec["reason"] and "846" in rec["reason"]


def test_ledger_index_flags_withdrawn_rows():
    rows = {r["exp_id"]: r for r in index()}
    r = rows["E0019-predictor-loss-math"]
    assert r["verdict"] == "PASS", "history must not be rewritten"
    assert r["withdrawn"] is True, "but it must be flagged"
    assert r["superseded_by"] == "E0021-enforced-math"


def test_final_claims_does_not_promote_a_withdrawn_result():
    """Scan FINAL-CLAIMS.md: a withdrawn id may appear only in a section that
    is explicitly about withdrawal."""
    text = CLAIMS.read_text()
    w = set(withdrawn())
    for exp in w:
        for line in text.splitlines():
            if exp in line:
                low = line.lower()
                assert any(k in low for k in ("withdraw", "superseded",
                                              "retract", "not verified")), (
                    f"{exp} cited in FINAL-CLAIMS without a withdrawal marker:"
                    f"\n  {line.strip()}")


def test_trap_fires_on_promotion():
    ok, _ = withdrawn_result_promotion(["E0021-enforced-math"],
                                       ["E0019-predictor-loss-math"])
    assert ok
    ok, detail = withdrawn_result_promotion(
        ["E0019-predictor-loss-math", "E0021-enforced-math"],
        ["E0019-predictor-loss-math"])
    assert not ok and "E0019" in detail


def test_missing_evidence_is_red():
    assert run_trap_checks({})["withdrawn_result_promotion"][0] is False
