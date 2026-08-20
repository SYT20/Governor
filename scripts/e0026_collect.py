#!/usr/bin/env python3
"""E0026 — collect execution feedback for every LiveCodeBench candidate.

The expensive half of E0026, kept separate from the analysis so the analysis can
be re-run without re-executing 4000 programs.

WHAT THIS MAY READ
    question text, starter_code, platform, difficulty
    public_test_cases          -- visible to anyone attempting the problem
    code_list                  -- the candidate being judged

WHAT THIS MAY NOT READ, AND DOES NOT
    private_test_cases         -- the hidden set
    metadata                   -- its EMPTINESS is exactly the label:
                                  1530 empty/PASS, 2470 non-empty/FAIL,
                                  zero off-diagonal over all 4000 samples
    graded_list                -- carried through to the output as the OUTCOME
                                  label, never as an input to any feature

The pilot (40 problems, 400 executions) recorded zero cases of a candidate
failing public tests while passing hidden ones, on all three platforms. Since
public tests are a subset of the judge's, that is what a correct runner must
show, and it is the check that validates the functional/leetcode wrapper.

Usage:
    python scripts/e0026_collect.py [--limit N] [--timeout 6.0]
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

from governor.execfeedback.publictests import evaluate  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
PUBLIC = ROOT / "results" / "lcb_public_tests.json"
OUT = ROOT / "results" / "E0026_feedback.json"
SUBMISSION = ("livecodebench/submissions",
              "Gemini-Pro-1.5 (May)/Scenario.codegeneration_10_0.2_eval_all.json")


def load_submissions() -> dict:
    from huggingface_hub import hf_hub_download
    p = hf_hub_download(SUBMISSION[0], SUBMISSION[1], repo_type="dataset")
    return {r["question_id"]: r for r in json.load(open(p))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="problems (0 = all)")
    ap.add_argument("--timeout", type=float, default=6.0)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    keep = json.loads(PUBLIC.read_text())
    subs = load_submissions()
    qids = sorted(keep)
    if args.limit:
        qids = qids[:args.limit]

    print(f"problems={len(qids)} timeout={args.timeout}s", flush=True)
    rows, t0 = [], time.perf_counter()

    for i, qid in enumerate(qids, 1):
        v, sub = keep[qid], subs[qid]
        for si, code in enumerate(sub["code_list"]):
            s = time.perf_counter()
            fb = evaluate(code, v["public"], platform=v["platform"],
                          starter_code=v["starter_code"], timeout_s=args.timeout)
            rows.append({
                "qid": qid, "sample": si,
                "platform": v["platform"], "difficulty": v["difficulty"],
                "graded": bool(sub["graded_list"][si]),      # OUTCOME, not a feature
                "tokens": None,
                **fb.features(),
                "n_tests": fb.n_tests,
                "wall_s": time.perf_counter() - s,
            })
        if i % 25 == 0 or i == len(qids):
            el = time.perf_counter() - t0
            print(f"  {i}/{len(qids)}  {el:>6.0f}s  eta {el/i*(len(qids)-i):>6.0f}s",
                  flush=True)

    pathlib.Path(args.out).write_text(json.dumps(rows))
    print(f"wrote {args.out}  rows={len(rows)}  "
          f"total={time.perf_counter()-t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
