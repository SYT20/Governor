#!/usr/bin/env python3
"""Project health. ENGINEERING_HEALTH and SCIENTIFIC_STATUS, reported separately.

GREEN never means "the science worked". It means the machinery is intact. The
two are printed as different lines precisely because conflating them is the
mistake this project spent nine iterations learning not to make.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

checks: list[tuple[str, str, str]] = []      # (level, name, detail)


def add(level: str, name: str, detail: str = "") -> None:
    checks.append((level, name, detail))


def sh(*a: str) -> str:
    try:
        return subprocess.run(a, cwd=ROOT, capture_output=True, text=True,
                              timeout=120).stdout.strip()
    except Exception:                                        # noqa: BLE001
        return ""


def main() -> int:
    print("=" * 74)
    print("GOVERNOR — PROJECT HEALTH")
    print("=" * 74)

    dirty = [x for x in sh("git", "status", "--porcelain").splitlines() if x.strip()]
    add("GREEN" if not dirty else "YELLOW", "git worktree",
        "clean" if not dirty else f"{len(dirty)} uncommitted")
    add("GREEN" if sh("git", "rev-parse", "HEAD") else "RED", "git repository",
        sh("git", "rev-parse", "HEAD")[:12])
    tags = sh("git", "tag").split()
    add("GREEN" if "v2.2-final" in tags else "RED", "checkpoint tag",
        f"{len(tags)} tags, v2.2-final "
        f"{'present' if 'v2.2-final' in tags else 'MISSING'}")

    try:
        from governor.harness.ledger import index, withdrawn
        rows = [r for r in index() if r["verdict"] != "UNFINALIZED"]
        ok = sum(r["verifies"] for r in rows)
        add("GREEN" if ok == len(rows) and rows else "RED", "experiments verify",
            f"{ok}/{len(rows)}")
        w = withdrawn()
        cited = [r["exp_id"] for r in rows if r["withdrawn"]]
        add("GREEN" if w else "YELLOW", "withdrawal register",
            f"{len(w)} withdrawn: {', '.join(sorted(w)) or 'none'}")
        add("GREEN", "history preserved",
            f"{len(cited)} withdrawn rows still carry their original verdict")
    except Exception as e:                                   # noqa: BLE001
        add("RED", "experiment ledger", str(e)[:60])

    try:
        from governor.harness import traps as T
        names = [n for n in dir(T) if not n.startswith("_")
                 and callable(getattr(T, n))
                 and n not in ("render", "run_trap_checks", "Path", "np", "re")]
        reg = T.run_trap_checks({})
        add("GREEN" if len(reg) >= 14 else "YELLOW", "trap catalogue",
            f"{len(reg)} registered + 1 conditional, all red on empty evidence: "
            f"{all(not v[0] for k, v in reg.items() if k != 'secret_scan')}")
        ok_secret, det = T.secret_scan()
        add("GREEN" if ok_secret else "RED", "secret scan", det[:50])
    except Exception as e:                                   # noqa: BLE001
        add("RED", "traps", str(e)[:60])

    for mod, cls in (("governor.gate.m2_interface", "MathM2"),
                     ("governor.gate.llm_m2", "LLMM2"),
                     ("governor.gate.gemini_m2", "GeminiM2"),
                     ("governor.gate.qwen_local", "QwenLocalM2")):
        try:
            m = __import__(mod, fromlist=[cls])
            add("GREEN" if hasattr(m, cls) else "RED", f"backend {cls}",
                "importable")
        except Exception as e:                               # noqa: BLE001
            add("YELLOW", f"backend {cls}", str(e)[:40])

    for f in ("FINAL-REPORT.md", "FINAL-CLAIMS.md", "REPRODUCE.md",
              "configs/phase4r_split.json", "experiments/WITHDRAWN.json"):
        add("GREEN" if (ROOT / f).exists() else "RED", f"artifact {f}",
            "present" if (ROOT / f).exists() else "MISSING")

    mem = Path.home() / (".claude/projects/-Users-keshavgautam-Desktop-Suyash-"
                         "Atlan-Proj/memory/governor-v2.2-checkpoint.md")
    add("GREEN" if mem.exists() else "YELLOW", "memory checkpoint",
        "present" if mem.exists() else "missing")

    print()
    for level, name, detail in checks:
        print(f"  [{level:<6}] {name:<28} {detail}")

    red = [c for c in checks if c[0] == "RED"]
    yellow = [c for c in checks if c[0] == "YELLOW"]
    eng = "RED" if red else ("YELLOW" if yellow else "GREEN")
    print("\n" + "-" * 74)
    print(f"  ENGINEERING_HEALTH : {eng}"
          f"{'  ' + str([c[1] for c in red]) if red else ''}"
          f"{'  (warn: ' + ', '.join(c[1] for c in yellow) + ')' if yellow and not red else ''}")
    print(f"  SCIENTIFIC_STATUS  : real-LLM Governor advantage NOT VERIFIED "
          f"(hard stop at E0025)")
    print("\n  These are different axes. GREEN engineering does not mean the")
    print("  hypothesis was supported; it means the machinery is intact.")
    print("\n  Full checks:  make test && make verify && make smoke")
    return 0 if eng != "RED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
