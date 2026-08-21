#!/usr/bin/env python3
"""Grade generated samples against LiveCodeBench's PRIVATE tests.

The missing stage. E0029 generated 4750 Qwen samples carrying only public-test
outcomes, so utility could only have been defined from the same signal the
features come from -- perfect leakage, and exactly what the preregistration
forbade after the published metadata turned out to BE the label.

This produces an independent hidden label:

    PUBLIC execution   ->  observable features  ->  Governor decision
    PRIVATE execution  ->  outcome label        ->  scoring, afterwards

Guarantees:

  IMMUTABLE INPUT   the generation file is opened read-only and never written.
                    Grades go to a separate file, joined by (problem_id, sample_id).
  RESUMABLE         every graded row is appended and flushed at each problem
                    boundary; a restart skips completed samples keyed on
                    (problem_id, sample_id) and refuses to overwrite a conflict.
  BOUNDED           min(8, cpu_count) workers via a SPAWN context. Fork is unsafe
                    after CUDA and deadlocked a run here for two hours.
  SURVIVABLE        one bad generated program cannot end the job. A crash, hang
                    or allocation storm is an outcome, recorded as such.

WORK IS BATCHED BY PROBLEM, not by sample. A problem's private tests are large
and all ~10 of its samples need the identical set; per-sample jobs would pickle
that suite ten times over and send it across the process boundary ten times.

Usage:
    python scripts/e0029_grade.py --limit 10        # smoke on real samples
    python scripts/e0029_grade.py --generations F   # grade an alternate file
    python scripts/e0029_grade.py                   # full run, resumable
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import os
import pathlib
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from governor.execfeedback.privatetests import (          # noqa: E402
    PrivateTestError, decode_private_tests, grade,
)

ROOT = pathlib.Path(__file__).resolve().parents[1]
GENERATIONS = [ROOT / "results" / "e0029_colab_generations.jsonl",
               ROOT / "results" / "E0029_QWEN.jsonl"]
OUT = ROOT / "results" / "E0029_QWEN_graded.jsonl"
PRIVATE_CACHE = ROOT / "results" / "e0029_private_tests.json"
DATASETS = ("test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl",
            "test5.jsonl", "test6.jsonl")

# Generation files have used both spellings for the same two columns.
ID_KEYS = ("problem_id", "qid")
SAMPLE_KEYS = ("sample_id", "sample")


def _pick(row: dict, keys: tuple[str, ...], default=None):
    for k in keys:
        if k in row:
            return row[k]
    return default


def load_generations(explicit: str | None) -> tuple[list[dict], pathlib.Path]:
    if explicit:
        src = pathlib.Path(explicit)
        if not src.is_absolute():
            src = ROOT / src
        if not src.exists():
            raise SystemExit(f"no such generation file: {src}")
    else:
        src = next((p for p in GENERATIONS if p.exists()), None)
        if src is None:
            raise SystemExit(
                "no E0029 generation file found. Expected one of:\n  "
                + "\n  ".join(str(p) for p in GENERATIONS)
                + "\n\nGeneration ran on the Colab VM; copy the raw JSONL back here,"
                  "\nor run this script there. Grading is CPU-only -- no GPU needed.")

    rows, torn, unusable = [], 0, 0
    for line in src.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            torn += 1                                     # interrupted final write
            continue
        qid, sid = _pick(d, ID_KEYS), _pick(d, SAMPLE_KEYS)
        if qid is None or sid is None:
            unusable += 1
            continue
        rows.append({"problem_id": qid, "sample_id": sid, "code": d.get("code", "")})
    if torn:
        print(f"  note: {torn} unparseable line(s) skipped", flush=True)
    if unusable:
        print(f"  note: {unusable} row(s) lack an id/sample column", flush=True)
    return rows, src


def resolve_private_tests(qids: set[str]) -> dict[str, dict]:
    """Fetch and decode private tests once, then cache. Records the source file."""
    if PRIVATE_CACHE.exists():
        try:
            cached = json.loads(PRIVATE_CACHE.read_text())
        except json.JSONDecodeError:
            cached = {}
        if cached and set(cached) >= qids:
            print(f"  private tests: {len(cached)} problems from cache", flush=True)
            return cached

    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    from huggingface_hub import hf_hub_download

    out: dict[str, dict] = {}
    failures: dict[str, str] = {}
    for fn in DATASETS:
        if set(out) >= qids:
            break
        try:
            p = hf_hub_download("livecodebench/code_generation_lite", fn,
                                repo_type="dataset")
        except Exception as e:                            # noqa: BLE001
            print(f"  {fn}: unavailable ({type(e).__name__})", flush=True)
            continue
        digest = hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()[:16]
        found = 0
        with open(p) as f:
            for line in f:
                if not line.strip():
                    continue
                d = json.loads(line)
                q = d.get("question_id")
                if q not in qids or q in out:
                    continue
                try:
                    tests = decode_private_tests(d["private_test_cases"])
                except (PrivateTestError, KeyError) as e:
                    failures[q] = str(e)
                    continue
                out[q] = {"tests": tests, "platform": d.get("platform", ""),
                          "starter_code": d.get("starter_code", ""),
                          "source_file": fn, "source_sha256_16": digest}
                found += 1
        print(f"  {fn}: +{found} problems (sha {digest})", flush=True)

    missing = sorted(qids - set(out))
    if missing:
        print(f"  WARNING: {len(missing)} problems have no private tests; "
              f"their samples are SKIPPED, not scored 0: {missing[:5]}", flush=True)
    if failures:
        print(f"  WARNING: {len(failures)} decode failures: "
              f"{list(failures.items())[:2]}", flush=True)
    PRIVATE_CACHE.write_text(json.dumps(out))
    return out


def _grade_problem(job: tuple) -> list[dict]:
    """Grade every pending sample of ONE problem. Module-level so it pickles."""
    qid, samples, tests, platform, starter, timeout_s, max_tests = job
    recs = []
    for sid, code in samples:
        try:
            g = grade(code, tests, platform=platform, starter_code=starter,
                      timeout_s=timeout_s, max_tests=max_tests or None)
            recs.append({"problem_id": qid, "sample_id": sid, **g.as_dict()})
        except Exception as e:                            # noqa: BLE001
            # A grader that dies on one program is worse than one that records it.
            n = len(tests[:max_tests] if max_tests else tests)
            recs.append({"problem_id": qid, "sample_id": sid,
                         "hidden_tests_total": n, "hidden_tests_passed": 0,
                         "hidden_tests_failed": n, "hidden_pass_fraction": 0.0,
                         "hidden_all_passed": False,
                         "grading_status": f"HARNESS_{type(e).__name__}",
                         "grading_latency_ms": 0})
    return recs


def done_keys(out: pathlib.Path) -> tuple[set, dict]:
    if not out.exists():
        return set(), {}
    keys, rows = set(), {}
    for line in out.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        k = (d["problem_id"], d["sample_id"])
        keys.add(k)
        rows[k] = d
    return keys, rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations", default=None,
                    help="path to the generation JSONL (default: autodetect)")
    ap.add_argument("--out", default=None,
                    help=f"output path (default: results/{OUT.name})")
    ap.add_argument("--limit", type=int, default=0, help="grade only N samples")
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    ap.add_argument("--timeout", type=float, default=6.0)
    ap.add_argument("--max-tests", type=int, default=0,
                    help="cap private tests per sample (0 = all)")
    args = ap.parse_args()

    out_path = pathlib.Path(args.out) if args.out else OUT
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows, src = load_generations(args.generations)
    print(f"generations: {len(rows)} rows from {src.name}")
    qids = {r["problem_id"] for r in rows}
    print(f"problems   : {len(qids)}")

    priv = resolve_private_tests(qids)
    have, existing = done_keys(out_path)
    print(f"already graded: {len(have)}")

    pending: dict[str, list] = defaultdict(list)
    budget = args.limit or None
    skipped_noprivate = 0
    for r in rows:
        if budget is not None and budget <= 0:
            break
        key = (r["problem_id"], r["sample_id"])
        if key in have:
            continue
        if r["problem_id"] not in priv:
            skipped_noprivate += 1
            continue
        pending[r["problem_id"]].append((r["sample_id"], r["code"]))
        if budget is not None:
            budget -= 1

    jobs = [(q, s, priv[q]["tests"], priv[q]["platform"], priv[q]["starter_code"],
             args.timeout, args.max_tests) for q, s in pending.items()]
    n_samples = sum(len(s) for s in pending.values())
    def n_tests(q: str) -> int:
        t = priv[q]["tests"]
        return min(len(t), args.max_tests) if args.max_tests else len(t)

    per = [n_tests(q) for q in pending]
    total_tests = sum(len(v) * n_tests(q) for q, v in pending.items())

    print(f"to grade   : {n_samples} samples over {len(jobs)} problems")
    if per:
        print(f"tests/prob : min {min(per)}  median {sorted(per)[len(per)//2]}  "
              f"max {max(per)}")
    print(f"executions : {total_tests:,}")
    if skipped_noprivate:
        print(f"skipped    : {skipped_noprivate} samples (no private tests)")
    print(f"workers    : {args.workers} (spawn)\n", flush=True)
    if not jobs:
        print("nothing to do")
        return 0

    t0 = time.perf_counter()
    failures = graded = conflicts = 0
    ctx = multiprocessing.get_context("spawn")
    with open(out_path, "a") as sink, ProcessPoolExecutor(max_workers=args.workers,
                                                     mp_context=ctx) as pool:
        futs = {pool.submit(_grade_problem, j): j[0] for j in jobs}
        for done_n, fut in enumerate(as_completed(futs), 1):
            for rec in fut.result():
                key = (rec["problem_id"], rec["sample_id"])
                if key in existing and existing[key] != rec:
                    conflicts += 1
                    continue
                sink.write(json.dumps(rec) + "\n")
                graded += 1
                if rec["grading_status"].startswith("HARNESS"):
                    failures += 1
            sink.flush()                                  # a kill loses <=1 problem
            if done_n % 25 == 0 or done_n == len(jobs):
                el = time.perf_counter() - t0
                eta = el / done_n * (len(jobs) - done_n)
                print(f"  {done_n}/{len(jobs)} problems  {graded} samples  "
                      f"{el/60:>5.1f}min  eta {eta/60:>5.1f}min  "
                      f"harness-failures {failures}", flush=True)

    el = time.perf_counter() - t0
    print(f"\ngraded {graded} samples in {el/60:.1f} min "
          f"({graded/max(el,1e-9):.1f}/s, {args.workers} workers)")
    print(f"harness failures: {failures}")
    if conflicts:
        print(f"CONFLICTS refused: {conflicts} (existing row differed; kept original)")
    print(f"wrote {out_path}")
    print(f"generation file NOT modified: {src.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
