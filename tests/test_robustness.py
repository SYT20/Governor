"""Operational failure modes: the system must fail LOUDLY, never into a PASS.

Every case here is a way a run can break in practice. The requirement is not
that nothing breaks -- it is that a broken run is visibly broken rather than
quietly producing a number someone later quotes.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pytest

from governor.harness.ledger import (
    ExperimentRun, ExperimentSpec, ProvenanceError, index, verify_experiment,
)
from governor.phase4.collect import CallRecord, ResponseCache, _key
from governor.phase4.config import PROMPT_CAP
from governor.phase4.env import P4Env, make_episodes
from governor.phase4.evaluate import constant, execute
from governor.phase4.policies import all_cheap, greedy
from governor.phase4.tasks import make_pool

LOW, HIGH, BUDGET = 300, 700, 4000.0


def _spec(exp_id="E9100"):
    return ExperimentSpec(
        exp_id=exp_id, title="robustness", model="synthetic",
        budget={"b": 1}, seeds={"s": 0}, split={"n": 1}, metric="none")


def _run(root, n=4, exp_id="E9100"):
    r = ExperimentRun(_spec(exp_id), root=root)
    for i in range(n):
        r.append({"i": i, "u": 0.5})
    return r


# -- interrupted / partial runs -------------------------------------------------

def test_a_killed_run_leaves_evidence_not_a_result(tmp_path):
    """Process dies mid-experiment: raw rows survive, results.json does not."""
    r = _run(tmp_path)
    r.flush()
    assert r.raw_path.exists() and r.raw_path.read_text().strip()
    assert not (r.dir / "results.json").exists()
    rows = index(root=tmp_path)
    assert rows[0]["verdict"] == "UNFINALIZED" and rows[0]["verifies"] is False


def test_abort_marks_the_directory(tmp_path):
    r = _run(tmp_path)
    r.abort("backend unreachable")
    assert (r.dir / "ABORTED.txt").exists()
    assert "backend unreachable" in (r.dir / "ABORTED.txt").read_text()
    assert not (r.dir / "results.json").exists()


def test_duplicate_experiment_id_is_refused(tmp_path):
    r = _run(tmp_path)
    r.finalize(summary={"u": 1}, metrics={}, verdict="PASS", allow_dirty=True)
    with pytest.raises(ProvenanceError, match="already finalized"):
        ExperimentRun(_spec(), root=tmp_path)


# -- corrupted / missing artifacts ----------------------------------------------

def test_corrupted_raw_file_fails_verification(tmp_path):
    r = _run(tmp_path)
    r.finalize(summary={"u": 1}, metrics={}, verdict="PASS", allow_dirty=True)
    r.raw_path.write_text(r.raw_path.read_text().replace("0.5", "0.9"))
    ok, bad = verify_experiment("E9100", root=tmp_path)
    assert not ok and any("modified" in b for b in bad)


def test_missing_metrics_file_fails_verification(tmp_path):
    r = _run(tmp_path)
    r.finalize(summary={"u": 1}, metrics={}, verdict="PASS", allow_dirty=True)
    (r.dir / "metrics.json").unlink()
    ok, bad = verify_experiment("E9100", root=tmp_path)
    assert not ok and any("metrics.json" in b for b in bad)


def test_truncated_raw_row_count_is_caught(tmp_path):
    r = _run(tmp_path, n=6)
    r.finalize(summary={"u": 1}, metrics={}, verdict="PASS", allow_dirty=True)
    lines = r.raw_path.read_text().splitlines()
    r.raw_path.write_text("\n".join(lines[:-2]) + "\n")
    ok, bad = verify_experiment("E9100", root=tmp_path)
    assert not ok


def test_a_verdict_inconsistent_with_its_traps_fails_verification(tmp_path):
    """A PASS carrying a red trap must not survive an audit months later."""
    r = _run(tmp_path)
    r.finalize(summary={"u": 1}, metrics={}, verdict="PASS", allow_dirty=True)
    res = json.loads((r.dir / "results.json").read_text())
    res["red_traps"] = ["budget_adherence"]
    (r.dir / "results.json").write_text(json.dumps(res))
    ok, bad = verify_experiment("E9100", root=tmp_path)
    assert not ok and any("red traps" in b for b in bad)


# -- cache integrity -------------------------------------------------------------

def test_cache_key_separates_everything_that_changes_semantics():
    base = dict(model="m", item_id="i1", prompt="p", max_tokens=700, temp=0.0)
    k0 = _key(**base)
    for field, other in (("model", "other-model"), ("item_id", "i2"),
                         ("prompt", "p2"), ("max_tokens", 2800), ("temp", 0.7)):
        assert _key(**{**base, field: other}) != k0, field


def test_a_cache_miss_raises_rather_than_calling_out(tmp_path):
    """The environment must never reach for the network mid-episode."""
    cache = ResponseCache(tmp_path / "c.sqlite", model="synthetic")
    pool = make_pool(seed=1, n=4)
    env = P4Env(cache, [pool], LOW, HIGH, BUDGET, PROMPT_CAP)
    with pytest.raises(KeyError, match="collect first"):
        env.step(env.reset(0), "H")


def test_different_models_do_not_share_cached_responses(tmp_path):
    p = tmp_path / "shared.sqlite"
    it = make_pool(seed=2, n=1)[0]
    a = ResponseCache(p, model="model-a")
    a.put(it, LOW, CallRecord(it.item_id, LOW, "1", "stop", 10, 10, 10, 20, 0.1), 1)
    b = ResponseCache(p, model="model-b")
    assert b.get(it, LOW) is None, "model-b read model-a's response"
    assert a.get(it, LOW) is not None


def test_a_changed_prompt_invalidates_its_cache_entry(tmp_path):
    """The defect that cost a day: a semantic change must not hit a stale row."""
    from governor.phase4.tasks import Item
    cache = ResponseCache(tmp_path / "c.sqlite", model="m")
    old = Item("x1", "Compute: 2 * 3", 6, 1, 0, "expr")
    new = Item("x1", "Compute: 2 * 4", 8, 1, 0, "expr")   # same id, new prompt
    cache.put(old, LOW, CallRecord("x1", LOW, "6", "stop", 5, 5, 5, 10, 0.1), 1)
    assert cache.get(new, LOW) is None, "stale response served for a new prompt"


# -- determinism -----------------------------------------------------------------

def _det_env(tmp_path, seed=3):
    cache = ResponseCache(tmp_path / "d.sqlite", model="synthetic")
    pool = make_pool(seed=seed, n=32)
    for j, it in enumerate(pool):
        for mt in (LOW, HIGH):
            ok = (j % 3 != 0) or mt == HIGH
            used = int(mt * 0.8)
            cache.put(it, mt, CallRecord(it.item_id, mt,
                                         str(it.answer) if ok else "0",
                                         "stop", 40, used, used, 40 + used, 0.1), 1)
    return P4Env(cache, make_episodes(pool, 8, 5), LOW, HIGH, BUDGET, PROMPT_CAP)


def test_execution_is_bit_for_bit_deterministic(tmp_path):
    env = _det_env(tmp_path)
    E = list(range(len(env.episodes)))
    a = execute(env, "g", constant(greedy(env)), E)
    b = execute(env, "g", constant(greedy(env)), E)
    assert a.modes == b.modes
    assert np.array_equal(a.U, b.U) and np.array_equal(a.spent, b.spent)


def test_episode_grouping_is_seed_controlled(tmp_path):
    pool = make_pool(seed=7, n=40)
    assert ([i.item_id for e in make_episodes(pool, 10, 42) for i in e]
            == [i.item_id for e in make_episodes(pool, 10, 42) for i in e])
    assert ([i.item_id for e in make_episodes(pool, 10, 42) for i in e]
            != [i.item_id for e in make_episodes(pool, 10, 43) for i in e])


def test_predictor_is_seed_controlled(tmp_path):
    from governor.phase4.predictor import ValuePredictor
    pool = make_pool(seed=8, n=60)
    g = np.array([i.n_ops for i in pool], float)
    a = ValuePredictor(kind="gbt", seed=0).fit(pool, g)
    b = ValuePredictor(kind="gbt", seed=0).fit(pool, g)
    assert a.cv_r2 == b.cv_r2


# -- backend isolation -------------------------------------------------------------

def test_governor_never_imports_a_specific_backend():
    """The Governor must not know which engine is behind the M2 contract."""
    import ast
    for mod in ("governor/phase4/policies.py", "governor/phase4/predictor.py",
                "governor/phase4/softbudget.py"):
        tree = ast.parse((Path(mod)).read_text())
        names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names.update(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                names.add(node.module)
        for n in names:
            for banned in ("llm_m2", "gemini_m2", "qwen_local", "openai", "groq"):
                assert banned not in n, f"{mod} imports {n}"


def test_every_backend_satisfies_the_m2_result_contract():
    from governor.gate.m2_interface import M2Result
    from governor.gate.gemini_m2 import GeminiM2
    from governor.gate.llm_m2 import LLMM2
    from governor.gate.m2_interface import MathM2
    from governor.gate.qwen_local import QwenLocalM2
    for cls in (MathM2, LLMM2, GeminiM2, QwenLocalM2):
        inst = cls() if cls is MathM2 else cls.__new__(cls)
        assert hasattr(inst, "name") or hasattr(cls, "name"), cls
        assert callable(getattr(cls, "__call__", None)), cls
    r = MathM2()({}, 1.0)
    assert isinstance(r, M2Result)
    for f in ("result", "reasoning_tokens", "total_tokens", "latency_s",
              "cost_units", "ok", "error"):
        assert hasattr(r, f), f
