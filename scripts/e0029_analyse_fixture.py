#!/usr/bin/env python3
"""Prove e0029_analyse.py's gates actually fire, using synthetic data of the
exact E0029 shape.

A guard nobody has watched fail is not a guard. Each case below constructs data
that SHOULD trip a specific gate, and the fixture fails if the gate stays quiet.

Case 1  E0028 fingerprint  400 problems -> split 207/193, the precise wrong
                           dataset that produced the invalid result. Gate 0.
Case 2  label leak         a feature wired straight to the hidden outcome.
                           Gate 1 must catch it via the mutation probe, not by
                           reading its name -- the name is deliberately innocent.
Case 3  zero ceiling       no problem is rescuable by a later sample. Gate 2.
Case 4  no signal          hidden label independent of everything observable.
                           Gate 3 must refuse to hand a noise ranker onward.
Case 5  real signal        label correlated with the public evidence. All gates
                           pass and a Governor advantage is reported.

Run:  python scripts/e0029_analyse_fixture.py
"""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import tempfile

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
PROBLEMS = ROOT / "results" / "e0029_problems.json"
ANALYSE = ROOT / "scripts" / "e0029_analyse.py"

CODE_OK = "import sys\nd=sys.stdin.read().split()\nprint(len(d))\n"


PLATFORM = {p["qid"]: p.get("platform", "?")
            for p in json.loads(PROBLEMS.read_text())}


def make_rows(qids, rng, *, signal=True, rescuable=True, leak=False,
              noise=0.18, coupling=0.55, n_samples=10):
    """Synthesise generation+grade rows with a controllable label mechanism."""
    gens, grades = [], []
    for q in qids:
        hidden_by_sample = {}
        # per-problem latent difficulty drives BOTH public evidence and the
        # hidden label when signal=True; when False the label is independent.
        skill = rng.beta(2, 2)
        solved_first = rng.random() < skill * 0.45
        can_rescue = rescuable and (rng.random() < (0.10 + coupling * skill))
        for s in range(n_samples):
            pub = float(np.clip(rng.normal(skill, noise), 0, 1))
            if s == 0:
                hidden = solved_first
            elif not solved_first and can_rescue:
                hidden = rng.random() < (skill * 0.30 if signal else 0.12)
            else:
                hidden = solved_first
            if not signal:
                hidden = rng.random() < 0.30
                pub = float(rng.random())

            g = {"experiment_id": "E0029-QWEN", "problem_id": q, "sample_id": s,
                 "model": "Qwen/Qwen3-1.7B", "model_revision": "abc123",
                 "seed": 1000 + s,
                 "prompt_tokens": 400, "completion_tokens": int(rng.integers(80, 600)),
                 "latency_ms": int(rng.integers(200, 4000)),
                 "generation_status": "ok", "code": CODE_OK,
                 "platform": PLATFORM.get(q, "?"),
                 "execution_status": "ok",
                 "compile_ok": 1.0, "runtime_error": 0.0, "timeout": 0.0,
                 "output_nonempty": 1.0,
                 "pub_passed": round(pub * 5), "pub_failed": 5 - round(pub * 5),
                 "pub_frac": pub, "pub_all_passed": 1.0 if pub >= 0.999 else 0.0,
                 "exec_latency_s": float(rng.random()),
                 "output_truncated": 0.0}
            g["total_tokens"] = g["prompt_tokens"] + g["completion_tokens"]
            hidden_by_sample[s] = bool(hidden)
            gens.append(g)
            grades.append({"problem_id": q, "sample_id": s,
                           "hidden_tests_total": 10,
                           "hidden_tests_passed": 10 if hidden else 3,
                           "hidden_tests_failed": 0 if hidden else 7,
                           "hidden_pass_fraction": 1.0 if hidden else 0.3,
                           "hidden_all_passed": bool(hidden),
                           "grading_status": "OK" if hidden else "WRONG_ANSWER",
                           "grading_latency_ms": 120})
    if leak:
        # Bake the DECISION label -- "does any LATER sample succeed" -- into an
        # innocently named observable column. This survives label mutation (it is
        # already stored) and passes any blocklist of names, so only the
        # separation probe can see it. Leaking a PAST attempt's outcome instead
        # would be inert: decision rows exist only where every past attempt
        # failed, so that column is constant and carries no future information.
        byq = {}
        for g in gens:
            byq.setdefault(g["problem_id"], []).append(g)
        lab = {(x["problem_id"], x["sample_id"]): x["hidden_all_passed"]
               for x in grades}
        for q, rows in byq.items():
            rows.sort(key=lambda r: r["sample_id"])
            for i, r in enumerate(rows):
                later = any(lab[(q, o["sample_id"])] for o in rows[i + 1:])
                r["exec_latency_s"] = 1.0 if later else 0.0
    return gens, grades


def run_case(name, qids, *, expect_gate, rng_seed, **kw):
    rng = np.random.default_rng(rng_seed)
    gens, grades = make_rows(qids, rng, **kw)
    tmp = pathlib.Path(tempfile.mkdtemp())
    gp, rp = tmp / "gen.jsonl", tmp / "graded.jsonl"
    gp.write_text("".join(json.dumps(x) + "\n" for x in gens))
    rp.write_text("".join(json.dumps(x) + "\n" for x in grades))

    # point the analyser at the fixture files without editing it
    env_stub = tmp / "run.py"
    env_stub.write_text(
        "import pathlib, sys\n"
        f"sys.path.insert(0, {str(ROOT)!r})\n"
        "import scripts.e0029_analyse as A\n"
        f"A.GEN = pathlib.Path({str(gp)!r})\n"
        f"A.GRADED = pathlib.Path({str(rp)!r})\n"
        f"A.AUDIT_OUT = pathlib.Path({str(tmp / 'audit.json')!r})\n"
        f"A.FROZEN = pathlib.Path({str(tmp / 'frozen.json')!r})\n"
        f"A.FROZEN_MODEL = pathlib.Path({str(tmp / 'frozen.pkl')!r})\n"
        "sys.argv = ['e0029_analyse', '--freeze']\n"
        "rc = A.main()\n"
        "if rc == 0:\n"
        "    sys.argv = ['e0029_analyse']\n"
        "    rc = A.main()\n"
        "raise SystemExit(rc)\n")
    p = subprocess.run([sys.executable, str(env_stub)], capture_output=True,
                       text=True, timeout=1200)
    out = p.stdout + p.stderr

    reached = [g for g in ("GATE 0", "GATE 1", "GATE 2", "GATE 3", "GATE 4")
               if g in out]
    failed = "GATE FAILED" in out
    last = reached[-1] if reached else "none"

    if expect_gate is None:
        ok = (not failed) and "GATE 4" in out
        detail = "all gates passed" if ok else f"stopped at {last}"
    else:
        ok = failed and last == expect_gate
        detail = (f"stopped at {last} as expected" if ok
                  else f"expected stop at {expect_gate}, got {last} "
                       f"(failed={failed})")
    print(f"  {'PASS' if ok else 'FAIL'}  {name:22s} {detail}")
    if not ok:
        print("\n".join("        " + l for l in out.splitlines()[-25:]))
    return ok, out


def main() -> int:
    problems = json.loads(PROBLEMS.read_text())
    qids = [p["qid"] for p in problems]
    print(f"\ne0029_analyse.py gate fixture -- {len(qids)} real problem ids\n")

    results = []
    results.append(run_case("E0028 fingerprint", qids[:400], expect_gate="GATE 0",
                            rng_seed=1)[0])
    results.append(run_case("label leak", qids, expect_gate="GATE 1",
                            rng_seed=2, leak=True)[0])
    results.append(run_case("zero ceiling", qids, expect_gate="GATE 2",
                            rng_seed=3, rescuable=False)[0])
    results.append(run_case("no signal", qids, expect_gate="GATE 3",
                            rng_seed=4, signal=False)[0])
    # Strong, unambiguous signal: the public evidence tracks the latent skill
    # closely and rescuability depends on it. The point is to exercise gate 4's
    # code path, not to discover whether a marginal effect is real.
    ok, out = run_case("real signal", qids, expect_gate=None, rng_seed=5,
                       noise=0.06, coupling=0.85)
    results.append(ok)

    if ok:
        for line in out.splitlines():
            if any(t in line for t in ("advantage", "ceiling ", "captured",
                                       "VERDICT", "observable ceiling", "RED",
                                       "operating point", "Governor  ",
                                       "best fixed", "blocked by")):
                print("        " + line.strip())

    n = sum(results)
    print(f"\n  {n}/{len(results)} gate cases behaved as specified\n")
    return 0 if n == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
