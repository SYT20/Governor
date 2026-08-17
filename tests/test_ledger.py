"""The ledger must REFUSE. Each test asserts a refusal that would have caught a
specific way this project lost the provenance of a number.
"""
from __future__ import annotations

import json
import time

import pytest

from governor.harness.ledger import (
    ExperimentRun, ExperimentSpec, ProvenanceError, index, load_experiment,
    verify_experiment,
)


def spec(exp_id="E9001", **kw):
    base = dict(
        exp_id=exp_id, title="unit test", model="test_model",
        budget={"tokens": 100}, seeds={"cal": 0, "test": 1},
        split={"cal": 10, "test": 10, "disjoint": True},
        metric="fraction of items answered correctly",
    )
    base.update(kw)
    return ExperimentSpec(**base)


@pytest.fixture
def root(tmp_path):
    return tmp_path / "experiments"


def _run(root, n_rows=3, **kw):
    r = ExperimentRun(spec(**kw), root=root)
    for i in range(n_rows):
        r.append({"episode": i, "utility": 0.5})
    return r


def test_refuses_summary_without_raw(root):
    """U=0.8247 with no per-episode rows behind it is the case this blocks."""
    r = ExperimentRun(spec(), root=root)
    with pytest.raises(ProvenanceError, match="raw file is empty"):
        r.finalize(summary={"U": 0.8247}, metrics={}, allow_dirty=True)


def test_refuses_empty_summary(root):
    r = _run(root)
    with pytest.raises(ProvenanceError, match="empty summary"):
        r.finalize(summary={}, metrics={}, allow_dirty=True)


def test_refuses_missing_required_field(root):
    r = _run(root)
    r.config["metric"] = ""            # simulate a spec that never defined it
    with pytest.raises(ProvenanceError, match="metric"):
        r.finalize(summary={"U": 1.0}, metrics={}, allow_dirty=True)


def test_refuses_stale_raw_file(root):
    """A raw file left over from a previous run must not be adopted."""
    r = _run(root)
    r.flush()
    stale = time.time() - 86_400
    import os
    os.utime(r.raw_path, (stale, stale))
    with pytest.raises(ProvenanceError, match="outside this run's window"):
        r.finalize(summary={"U": 1.0}, metrics={}, allow_dirty=True)


def test_refuses_foreign_nonce_rows(root):
    r = _run(root)
    r.flush()
    with r.raw_path.open("a") as f:     # a row from somewhere else
        f.write(json.dumps({"episode": 99, "_nonce": "deadbeef"}) + "\n")
    r.n_rows += 1
    with pytest.raises(ProvenanceError, match="nonce"):
        r.finalize(summary={"U": 1.0}, metrics={}, allow_dirty=True)


def test_refuses_edited_config(root):
    r = _run(root)
    (r.dir / "config.json").write_text(json.dumps({"budget": "changed"}))
    with pytest.raises(ProvenanceError, match="config.json on disk differs"):
        r.finalize(summary={"U": 1.0}, metrics={}, allow_dirty=True)


def test_refuses_moved_head(root):
    r = _run(root)
    (r.dir / "git_commit.txt").write_text("0" * 40 + "\n")
    with pytest.raises(ProvenanceError, match="HEAD moved"):
        r.finalize(summary={"U": 1.0}, metrics={}, allow_dirty=True)


def test_red_trap_forces_blocked_verdict(root):
    """The caller does not get to declare PASS over a red check. In every
    historical case the caller was me and PASS had already printed."""
    r = _run(root)
    res = r.finalize(summary={"U": 1.0}, metrics={"delta": 0.03},
                     traps={"greedy_collapse": (False, "identical calls")},
                     verdict="PASS", allow_dirty=True)
    assert res["verdict"] == "BLOCKED"
    assert res["red_traps"] == ["greedy_collapse"]


def test_green_traps_keep_verdict(root):
    r = _run(root)
    res = r.finalize(summary={"U": 1.0}, metrics={},
                     traps={"secret_scan": (True, "clean")},
                     verdict="PASS", allow_dirty=True)
    assert res["verdict"] == "PASS"


def test_finalize_writes_all_six_files_and_verifies(root):
    r = _run(root, n_rows=5)
    r.finalize(summary={"U": 0.8}, metrics={"delta": 0.03}, verdict="PASS",
               allow_dirty=True)
    for f in ("config.json", "results.json", "metrics.json", "raw.jsonl",
              "git_commit.txt", "README.md"):
        assert (r.dir / f).exists(), f
    ok, bad = verify_experiment("E9001", root=root)
    assert ok, bad
    assert load_experiment("E9001", root=root)["summary"]["U"] == 0.8


def test_verify_detects_post_hoc_raw_edit(root):
    """The check that makes the ledger worth having: run it months later."""
    r = _run(root, n_rows=5)
    r.finalize(summary={"U": 0.8}, metrics={}, verdict="PASS", allow_dirty=True)
    with r.raw_path.open("a") as f:
        f.write(json.dumps({"episode": 5, "utility": 1.0,
                            "_nonce": r.nonce}) + "\n")
    ok, bad = verify_experiment("E9001", root=root)
    assert not ok
    assert any("modified since finalization" in b for b in bad)


def test_cannot_overwrite_a_finalized_experiment(root):
    r = _run(root)
    r.finalize(summary={"U": 0.8}, metrics={}, verdict="PASS", allow_dirty=True)
    with pytest.raises(ProvenanceError, match="already finalized"):
        ExperimentRun(spec(), root=root)


def test_index_reports_unfinalized_runs(root):
    _run(root, exp_id="E9002")          # opened, never finalized
    rows = {r["exp_id"]: r for r in index(root=root)}
    assert rows["E9002"]["verdict"] == "UNFINALIZED"
    assert rows["E9002"]["verifies"] is False
