#!/usr/bin/env python3
"""Orientation for a future session. Reads state; starts nothing.

Memory can drift from the repository. This reports what the repository actually
contains, so a resuming agent never has to trust a remembered number.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FROZEN = """FROZEN — do not change without new empirical evidence:
    the scientific conclusion below
    env6-reference values (0.6896875 / 0.813125 / 0.83875 at 1e-12)
    governor/gate/executor.py  (canonical executor)
    M2(state, reasoning_budget) -> M2Result
    thresholds: materiality 0.02, S1 headroom 0.12, S2 decidability 0.70
    frozen splits (configs/phase4r_split.json; doc_id and question-id parity)
    historical ledger records -- append withdrawals, never rewrite a verdict"""

CLOSED = """CLOSED — do not reopen:
    hard worst-case reservation      binds no, headroom yes
    forced Wait units (MATH, GPQA)   binds yes, headroom no
    MATH-500 as the settling set     needs ~26,031 items, has 500
    "another benchmark" / "more samples" as a route to significance"""

ALLOWED = """A NEW EXPERIMENT REQUIRES A GENUINELY NEW OBSERVABLE SIGNAL, e.g.
    verifier / test-execution feedback
    uncertainty or self-consistency structure beyond 2-sample agreement
    intermediate tool results or state transitions
    richer reasoning-trajectory features
Preregister hypothesis, ceiling, budget, split, pass and stop criteria.
Order is always: ceiling -> predictor -> controller. Never reversed."""


def sh(*a: str) -> str:
    try:
        return subprocess.run(a, cwd=ROOT, capture_output=True, text=True,
                              timeout=60).stdout.strip()
    except Exception:                                        # noqa: BLE001
        return "?"


def main() -> int:
    print("=" * 78)
    print("GOVERNOR — PROJECT RESUME")
    print("=" * 78)
    print(f"\n  commit   {sh('git', 'rev-parse', 'HEAD')[:12]}  "
          f"{sh('git', 'log', '-1', '--format=%s')[:52]}")
    print(f"  tags     {', '.join(sh('git', 'tag').split()[-4:])}")
    dirty = [x for x in sh("git", "status", "--porcelain").splitlines() if x.strip()]
    print(f"  worktree {'clean' if not dirty else f'{len(dirty)} uncommitted paths'}")

    try:
        from governor.harness.ledger import index
        rows = [r for r in index() if r["verdict"] != "UNFINALIZED"]
        ok = sum(r["verifies"] for r in rows)
        print(f"\n  experiments {len(rows)} finalized, {ok} verify"
              f"{'' if ok == len(rows) else '   <-- MISMATCH'}")
        from collections import Counter
        for v, c in sorted(Counter(r["verdict"] for r in rows).items()):
            print(f"      {v:<26}{c}")
    except Exception as e:                                   # noqa: BLE001
        print(f"  experiments: ledger unreadable ({str(e)[:60]})")

    print("\n  M2 backends")
    import importlib
    for mod, cls in (("governor.gate.m2_interface", "MathM2"),
                     ("governor.gate.llm_m2", "LLMM2"),
                     ("governor.gate.gemini_m2", "GeminiM2"),
                     ("governor.gate.qwen_local", "QwenLocalM2")):
        try:
            m = importlib.import_module(mod)
            print(f"      {cls:<14}{'importable' if hasattr(m, cls) else 'MISSING'}")
        except Exception as e:                               # noqa: BLE001
            print(f"      {cls:<14}import error: {str(e)[:40]}")

    print("\n" + "-" * 78)
    print("  SCIENTIFIC STATUS: real-LLM Governor advantage = NOT VERIFIED")
    print("    axis                       ceiling    Governor")
    for a, c, g in (("MATH tokens", "+0.164", "+0.0121 [-0.0396, +0.0510]"),
                    ("GPQA tokens", "+0.232", "+0.0000"),
                    ("LiveCodeBench samples", "+0.055", "-0.0028 [-0.0066, +0.0000]"),
                    ("LCB samples + probe", "+0.057", "+0.0046 [-0.0047, +0.0171]")):
        print(f"    {a:<27}{c:<11}{g}")
    print("\n  HARD STOP: E0025, binding. Recorded BLOCKED by the ledger after a")
    print("  red budget_adherence trap overrode the caller's verdict.")
    print("\n  ENGINEERING: complete. Run scripts/project_health.py for detail.")
    print("\n" + "-" * 78)
    print(FROZEN)
    print("\n" + CLOSED)
    print("\n" + ALLOWED)
    print("\n  Verify before trusting any of this:")
    print("      make test && make verify && make smoke")
    print("      python scripts/project_health.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
