"""The sequential protocol must be enforced, not promised."""
from __future__ import annotations

import json

import pytest

from governor.phase4 import gatekeeper as gk


def _write(tmp, exp_id, verdict, held_out=None, rows=1):
    d = tmp / exp_id
    d.mkdir(parents=True)
    (d / "results.json").write_text(json.dumps({
        "exp_id": exp_id, "verdict": verdict,
        "summary": {"held_out": held_out}, "git_commit": "a" * 40,
        "config_hash": "x", "nonce": "n", "raw_file": "raw.jsonl",
        "raw_sha256": "s", "raw_rows": rows, "red_traps": []}))
    return d


def test_refuses_when_the_gate_has_never_run(monkeypatch, tmp_path):
    monkeypatch.setattr(gk, "EXPERIMENTS", tmp_path)
    with pytest.raises(gk.GateNotPassed, match="has not been run"):
        gk.require_gate_passed()


def test_refuses_on_an_inconclusive_verdict(monkeypatch, tmp_path):
    monkeypatch.setattr(gk, "EXPERIMENTS", tmp_path)
    _write(tmp_path, gk.GATE_EXP, "GATE-INCONCLUSIVE-NEED-ITEMS")
    monkeypatch.setattr(gk, "verify_experiment", lambda e: (True, []))
    monkeypatch.setattr(gk, "load_experiment",
                        lambda e: json.loads((tmp_path / e / "results.json").read_text()))
    with pytest.raises(gk.GateNotPassed, match="INCONCLUSIVE"):
        gk.require_gate_passed()


def test_refuses_a_pass_that_had_no_held_out_items(monkeypatch, tmp_path):
    """An in-selection ceiling is not the gate, however good it looks."""
    monkeypatch.setattr(gk, "EXPERIMENTS", tmp_path)
    _write(tmp_path, gk.GATE_EXP, gk.PASS_VERDICT, held_out=None)
    monkeypatch.setattr(gk, "verify_experiment", lambda e: (True, []))
    monkeypatch.setattr(gk, "load_experiment",
                        lambda e: json.loads((tmp_path / e / "results.json").read_text()))
    with pytest.raises(gk.GateNotPassed, match="no held-out items"):
        gk.require_gate_passed()


def test_refuses_when_the_record_no_longer_verifies(monkeypatch, tmp_path):
    monkeypatch.setattr(gk, "EXPERIMENTS", tmp_path)
    _write(tmp_path, gk.GATE_EXP, gk.PASS_VERDICT, held_out={"ci_lo": 0.1})
    monkeypatch.setattr(gk, "verify_experiment",
                        lambda e: (False, ["raw.jsonl modified"]))
    monkeypatch.setattr(gk, "load_experiment",
                        lambda e: json.loads((tmp_path / e / "results.json").read_text()))
    with pytest.raises(gk.GateNotPassed, match="no longer verifies"):
        gk.require_gate_passed()


def test_allows_a_real_held_out_pass(monkeypatch, tmp_path):
    monkeypatch.setattr(gk, "EXPERIMENTS", tmp_path)
    _write(tmp_path, gk.GATE_EXP, gk.PASS_VERDICT,
           held_out={"ci_lo": 0.11, "n_items": 24})
    monkeypatch.setattr(gk, "verify_experiment", lambda e: (True, []))
    monkeypatch.setattr(gk, "load_experiment",
                        lambda e: json.loads((tmp_path / e / "results.json").read_text()))
    s = gk.require_gate_passed()
    assert s["ci_lo"] == 0.11 and s["n_items"] == 24


def test_the_real_gate_currently_blocks_everything_downstream():
    """Live check against the actual experiment directory: today it must refuse."""
    s = gk.gate_status()
    if s["verdict"] == gk.PASS_VERDICT:
        pytest.skip("gate has passed; this assertion no longer applies")
    with pytest.raises(gk.GateNotPassed):
        gk.require_gate_passed()
