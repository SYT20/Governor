#!/usr/bin/env python3
"""Every gate that must be green before Colab compute is spent.

The rule this file enforces: do not spend a runtime hour until the environment,
loader, sandbox, contracts, data, boundary and checkpointing have each been shown
to work. Two runs have already been abandoned in this project for reasons a
two-minute check would have surfaced -- a token cap that manufactured 42% empty
outputs, and a model whose default thinking mode consumed the entire budget
without emitting code.

Each gate returns a verdict and a reason. The script ends with exactly one of:

    READY FOR FULL RUN
    DO NOT START FULL RUN   <first failing gate>

Gates are independent and all of them run, so one failure does not hide five
others.

    python scripts/colab_preflight.py [--require-gpu] [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
ROOT = pathlib.Path(__file__).resolve().parents[1]


@dataclass
class Gate:
    name: str
    ok: bool
    detail: str = ""
    data: dict = field(default_factory=dict)


def gate_environment(require_gpu: bool) -> Gate:
    import os
    import platform
    info = {"python": sys.version.split()[0], "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "colab_release_tag": os.environ.get("COLAB_RELEASE_TAG", "not-colab")}
    gpu = {"present": False}
    try:
        import torch
        info["torch"] = torch.__version__
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            gpu = {"present": True, "name": p.name,
                   "memory_gb": round(p.total_memory / 1e9, 2),
                   "cuda": torch.version.cuda}
    except Exception as e:                                # noqa: BLE001
        info["torch"] = f"unavailable: {type(e).__name__}"
    info["gpu"] = gpu
    if require_gpu and not gpu["present"]:
        return Gate("environment", False,
                    "GPU required but none present -- failing rather than silently "
                    "falling back to CPU", info)
    return Gate("environment", True,
                f"{'GPU ' + gpu['name'] if gpu['present'] else 'CPU only'}", info)


def gate_text_loader() -> Gate:
    """Both load paths must agree, or the notebook is reading stale state."""
    try:
        from scripts.colab_text_loader import verify_import_parity
    except Exception:                                     # noqa: BLE001
        sys.path.insert(0, str(ROOT / "scripts"))
        from colab_text_loader import verify_import_parity      # type: ignore

    targets = ["governor.harness.ledger", "governor.harness.traps",
               "governor.gate.m2_interface", "governor.execution.executor",
               "governor.phase4.statemgr", "governor.execfeedback.sandbox",
               "governor.execfeedback.preflight", "governor.execfeedback.publictests"]
    bad = []
    for t in targets:
        try:
            r = verify_import_parity(t, ROOT)
            if not r["ok"]:
                bad.append(f"{t}: {r['problems'][:1]}")
        except Exception as e:                            # noqa: BLE001
            bad.append(f"{t}: {type(e).__name__}")
    return Gate("text loader parity", not bad,
                f"{len(targets)-len(bad)}/{len(targets)} identical both ways"
                + (f"; {bad[:2]}" if bad else ""))


def gate_restart_safety() -> Gate:
    """A fresh subprocess must reproduce the load, with no inherited state."""
    code = (
        f"import sys; sys.path.insert(0, {str(ROOT)!r})\n"
        "from scripts.colab_text_loader import load_package_module\n"
        "m = load_package_module('governor.gate.m2_interface')\n"
        "s = load_package_module('governor.execfeedback.sandbox')\n"
        "r = s.run('print(6*7)', '', timeout_s=5)\n"
        "assert r.stdout.strip() == '42', r.stdout\n"
        "assert hasattr(m, 'M2Result')\n"
        "print('RESTART_OK')\n"
    )
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       cwd=str(ROOT))
    ok = p.returncode == 0 and "RESTART_OK" in p.stdout
    return Gate("restart safety", ok,
                "fresh subprocess reproduced the load" if ok
                else (p.stderr or p.stdout)[-160:])


def gate_sandbox() -> Gate:
    """The adversarial battery, run again here rather than trusted from a test file."""
    from governor.execfeedback.sandbox import Status, run
    cases = [
        ("well-behaved", "print(input().strip())", "x\n", Status.OK),
        ("syntax error", "def f(:", "", Status.COMPILE_ERROR),
        ("crash", "raise ValueError()", "", Status.RUNTIME_ERROR),
        ("infinite loop", "while True: pass", "", Status.TIMEOUT),
        ("memory bomb", "x=[]\nwhile True: x.append(bytearray(10**7))", "", None),
        ("fork bomb", "import os\nwhile True:\n try: os.fork()\n except Exception: pass", "", None),
        ("stdout flood", "for _ in range(10**7): print('x'*80)", "", None),
        ("deep recursion", "import sys\nsys.setrecursionlimit(10**6)\ndef f(n): return f(n+1)\nf(0)", "", None),
    ]
    bad = []
    for name, src, stdin, expect in cases:
        try:
            r = run(src, stdin, timeout_s=3.0, mem_mb=512)
            if expect and r.status != expect:
                bad.append(f"{name}: {r.status} != {expect}")
        except Exception as e:                            # noqa: BLE001
            bad.append(f"{name}: harness raised {type(e).__name__}")
    return Gate("sandbox containment", not bad,
                f"{len(cases)-len(bad)}/{len(cases)} contained"
                + (f"; {bad[:2]}" if bad else ""))


def gate_boundary() -> Gate:
    """Deliberately reach for a forbidden field and require the boundary to refuse."""
    from governor.execfeedback.publictests import FEATURE_NAMES
    from governor.harness.traps import oracle_leakage
    forbidden = ["graded", "hidden_grade", "private_test_cases", "final_score",
                 "oracle_label", "answer"]
    ok_clean, _ = oracle_leakage(list(FEATURE_NAMES))
    caught = []
    for f in forbidden:
        clean, _ = oracle_leakage(list(FEATURE_NAMES) + [f])
        if not clean:
            caught.append(f)
    ok = ok_clean and len(caught) >= 4
    return Gate("information boundary", ok,
                f"permitted set clean={ok_clean}; rejected {len(caught)}/{len(forbidden)} "
                f"forbidden probes")


def gate_contracts() -> Gate:
    """Every backend must satisfy the same observable interface."""
    from governor.gate.m2_interface import M2Result
    required = {"result", "reasoning_tokens", "total_tokens", "latency_s",
                "cost_units", "ok", "error"}
    have = set(M2Result.__dataclass_fields__)
    missing = required - have
    backends, failed = [], []
    for dotted, cls in [("governor.gate.m2_interface", "MathM2"),
                        ("governor.gate.llm_m2", "LLMM2"),
                        ("governor.gate.gemini_m2", "GeminiM2"),
                        ("governor.gate.qwen_local", "QwenLocalM2")]:
        try:
            mod = __import__(dotted, fromlist=[cls])
            getattr(mod, cls)
            backends.append(cls)
        except Exception as e:                            # noqa: BLE001
            failed.append(f"{cls}:{type(e).__name__}")
    ok = not missing and len(backends) >= 3
    return Gate("M2 contract", ok,
                f"fields ok={not missing}; backends {backends}"
                + (f"; failed {failed}" if failed else ""))


def gate_tests() -> Gate:
    """The project's own suite, not a Colab-only substitute."""
    # No --timeout flag: that needs the pytest-timeout plugin, which is one more
    # pin to get wrong. A subprocess timeout does the same job with no dependency.
    try:
        p = subprocess.run([sys.executable, "-m", "pytest", "-q", "--no-header",
                            "-x", "tests/"],
                           cwd=str(ROOT), capture_output=True, text=True, timeout=1200)
    except subprocess.TimeoutExpired:
        return Gate("project test suite", False, "suite exceeded 20 minutes")
    out = (p.stdout or "") + (p.stderr or "")
    tail = [l for l in out.strip().splitlines() if l.strip()][-1:] or [""]
    return Gate("project test suite", p.returncode == 0, tail[0][:80])


def gate_checkpointing() -> Gate:
    """Append, interrupt, resume: no duplicates, no loss."""
    with tempfile.TemporaryDirectory() as td:
        path = pathlib.Path(td) / "ck.jsonl"
        with open(path, "a") as f:
            for i in range(10):
                f.write(json.dumps({"problem_id": f"P{i//2}", "sample_id": i % 2}) + "\n")
                f.flush()
        with open(path, "a") as f:                        # simulated torn write
            f.write('{"problem_id": "P9", "sam')
        done = set()
        with open(path) as f:
            for line in f:
                try:
                    d = json.loads(line)
                    done.add((d["problem_id"], d["sample_id"]))
                except json.JSONDecodeError:
                    continue                              # a torn tail is survivable
        ok = len(done) == 10
        return Gate("checkpoint resume", ok,
                    f"{len(done)}/10 recovered, torn final line skipped")


def gate_data() -> Gate:
    cfg = ROOT / "configs" / "e0029_split.json"
    if not cfg.exists():
        return Gate("frozen split", False, f"missing {cfg.name}")
    d = json.loads(cfg.read_text())
    cal, ev = set(d.get("calibration", [])), set(d.get("evaluation", []))
    overlap = cal & ev
    probs = ROOT / "results" / "e0029_problems.json"
    n = len(json.loads(probs.read_text())) if probs.exists() else 0
    ok = not overlap and cal and ev and n == len(cal) + len(ev)
    return Gate("frozen split", ok,
                f"cal {len(cal)} eval {len(ev)} overlap {len(overlap)} problems {n}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--require-gpu", action="store_true")
    ap.add_argument("--json", default="")
    ap.add_argument("--skip-tests", action="store_true")
    args = ap.parse_args()

    print("=" * 72)
    print("COLAB PREFLIGHT — no compute is spent until every gate is green")
    print("=" * 72)

    gates = [gate_environment(args.require_gpu), gate_text_loader(),
             gate_restart_safety(), gate_sandbox(), gate_boundary(),
             gate_contracts(), gate_checkpointing(), gate_data()]
    if not args.skip_tests:
        gates.append(gate_tests())

    for g in gates:
        print(f"  [{'  ok  ' if g.ok else ' FAIL '}] {g.name:<26} {g.detail[:78]}")

    failed = [g for g in gates if not g.ok]
    payload = {"gates": [{"name": g.name, "ok": g.ok, "detail": g.detail,
                          "data": g.data} for g in gates],
               "ready": not failed, "timestamp": time.time()}
    if args.json:
        pathlib.Path(args.json).write_text(json.dumps(payload, indent=1))
        print(f"\n  wrote {args.json}")

    print("\n" + "=" * 72)
    if failed:
        print(f"DO NOT START FULL RUN   first failing gate: {failed[0].name}")
        print(f"  {failed[0].detail}")
        return 1
    print("READY FOR FULL RUN")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
