#!/usr/bin/env python3
"""PHASE 3 — the fixed-budget reasoning curve. One question only:
does LLMM2 have a controllable value-vs-budget curve?

Do NOT touch the Governor until this is measured. Items span two difficulties so
the curve can show that harder items need more budget -- the property the
Governor would later exploit.
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("OR_KEY", "OPENROUTER_API_KEY_REDACTED")
from governor.gate.llm_m2 import LLMM2

EASY = [(a, b) for a, b in [(23, 14), (31, 12), (42, 11), (25, 13), (34, 21)]]
HARD = [(a, b) for a, b in [(47, 68), (83, 76), (59, 87), (94, 63), (78, 49)]]
BUDGETS = [200, 500, 900, 1500]


def run(m, items, budget, tag):
    ok = corr = 0; rt = 0.0
    for a, b in items:
        r = m({"prompt": f"What is {a}*{b}?"}, budget)
        rt += r.reasoning_tokens
        if r.ok:
            ok += 1
            try:
                corr += int(int("".join(c for c in str(r.result) if c.isdigit() or c == '-')) == a * b)
            except ValueError:
                pass
    n = len(items)
    print(f"    {tag:<5} b={budget:<5} answered {ok}/{n}  correct {corr}/{n}"
          f"  utility {corr/n:.2f}  mean reasoning tokens {rt/n:.0f}", flush=True)
    return {"tag": tag, "budget": budget, "answered": ok, "correct": corr,
            "n": n, "utility": corr / n, "mean_reasoning_tokens": rt / n}


def main() -> int:
    m = LLMM2()
    print("PHASE 3 — LLMM2 fixed-budget reasoning curve (nemotron-nano-9b-v2)")
    rows = []
    for b in BUDGETS:
        rows.append(run(m, EASY, b, "easy"))
        rows.append(run(m, HARD, b, "hard"))
    Path("results/llm_m2_curve.json").write_text(json.dumps(rows, indent=2))
    e = {r["budget"]: r["utility"] for r in rows if r["tag"] == "easy"}
    h = {r["budget"]: r["utility"] for r in rows if r["tag"] == "hard"}
    rises = any(h[b2] > h[b1] for b1, b2 in zip(BUDGETS, BUDGETS[1:]))
    gap = any(e[b] > h[b] for b in BUDGETS)
    print(f"\n  utility rises with budget on HARD items : {rises}")
    print(f"  easy items reach utility at lower budget : {gap}")
    print(f"\n  PHASE 3: {'PASS -- controllable value-vs-budget curve exists'
                          if (rises and gap) else 'FAIL -- diagnose M2, do not touch Governor'}")
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
