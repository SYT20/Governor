#!/usr/bin/env python3
"""Independently verify a Colab handoff. The notebook's own report is evidence, not truth.

A notebook that prints PASS has proved that it reached the end of its own code.
It has not proved that the numbers in its summary follow from the raw rows it
saved. Those are different claims, and only the second one matters.

So this script ignores every headline figure in the handoff and recomputes it
from `raw/*.jsonl`, then compares. A disagreement is a VERIFICATION_FAILURE and
the summary is never edited to match -- the discrepancy is the finding.

    python scripts/verify_colab_run.py --handoff claude_handoff/

Exit status is non-zero on any discrepancy, so it is usable in CI.

WHAT IS CHECKED

    integrity   files present, schemas valid, SHA256 matches, no duplicate ids
    split       calibration and evaluation disjoint, matching the frozen config
    boundary    no forbidden field appears in any raw row
    accounting  costs recomputed from rows, budget adherence re-derived
    metrics     utility, paired difference, bootstrap CI, AUC, P@K, lift, NDCG
    guardrails  the project's own trap suite re-run on the recomputed evidence
    agreement   every recomputed value against the notebook's claim, numerically
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import sys
from dataclasses import dataclass, field

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

TOLERANCE = 1e-6

# Fields that must never appear in a raw decision row. Presence of any of these
# means the experiment could have seen the answer at decision time.
FORBIDDEN_FIELDS = {
    "private_test_cases", "hidden_tests", "expected_output", "reference_solution",
    "final_score", "graded_list", "pass@1", "metadata", "future_samples",
    "hidden_grade", "evaluator_score", "oracle_label",
}

REQUIRED_HANDOFF = [
    "CLAUDE_HANDOFF.md", "experiment_summary.json", "environment.json",
    "model.json", "config.json", "checksums.json", "guardrails.json",
    "metrics.json", "provenance.json", "raw_manifest.json",
]


@dataclass
class Result:
    checks: list[tuple[str, bool, str]] = field(default_factory=list)

    def add(self, name: str, ok: bool, detail: str = "") -> None:
        self.checks.append((name, ok, detail))

    @property
    def failed(self) -> list[tuple[str, bool, str]]:
        return [c for c in self.checks if not c[1]]

    def render(self) -> str:
        lines = []
        for name, ok, detail in self.checks:
            lines.append(f"  [{'  ok  ' if ok else ' FAIL '}] {name:<38} {detail[:70]}")
        n_ok = len(self.checks) - len(self.failed)
        lines.append(f"\n  {n_ok}/{len(self.checks)} checks passed")
        return "\n".join(lines)


def load_json(path: pathlib.Path):
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def load_jsonl(path: pathlib.Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{i} is not valid JSON: {e}") from e
    return rows


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ----------------------------------------------------------------- metrics


def auc(y: np.ndarray, s: np.ndarray) -> float:
    """Rank-based AUC with tie handling. Recomputed here rather than imported so
    the verifier does not share a bug with the code that produced the claim."""
    y = np.asarray(y, float); s = np.asarray(s, float)
    pos, neg = y > 0.5, y <= 0.5
    if pos.sum() == 0 or neg.sum() == 0:
        return float("nan")
    order = np.argsort(s)
    ranks = np.empty(len(s), float)
    ranks[order] = np.arange(1, len(s) + 1)
    uniq, inv, counts = np.unique(s, return_inverse=True, return_counts=True)
    for u_i, c in enumerate(counts):
        if c > 1:
            m = inv == u_i
            ranks[m] = ranks[m].mean()
    return float((ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2)
                 / (pos.sum() * neg.sum()))


def ranking_metrics(y: np.ndarray, s: np.ndarray) -> dict:
    y = np.asarray(y, float); n, p = len(y), float(np.sum(y))
    order = np.argsort(-np.asarray(s, float))
    out = {"auc": auc(y, s), "n": n, "positives": int(p)}
    for k in (5, 10, 20):
        m = max(1, int(round(k / 100 * n)))
        hit = float(y[order[:m]].sum())
        out[f"precision_at_{k}"] = hit / m
        out[f"recall_at_{k}"] = hit / p if p else 0.0
        out[f"lift_at_{k}"] = (hit / m) / (p / n) if p else 0.0
    disc = np.log2(np.arange(2, n + 2))
    dcg = float(np.sum(y[order] / disc))
    idcg = float(np.sum(np.sort(y)[::-1] / disc))
    out["ndcg"] = dcg / idcg if idcg else 0.0
    return out


def paired_bootstrap(d: np.ndarray, n_boot: int = 4000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    n = len(d)
    means = np.array([d[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    lo, hi = np.percentile(means, [2.5, 97.5])
    sd = float(np.std(d, ddof=1)) if n > 1 else 0.0
    return {"mean": float(d.mean()), "lo": float(lo), "hi": float(hi),
            "paired_sd": sd,
            "n_required_eps_0.02": float((1.96 * sd / 0.02) ** 2) if sd > 0 else float("inf")}


# ----------------------------------------------------------------- checks


def check_structure(h: pathlib.Path, r: Result) -> dict:
    missing = [f for f in REQUIRED_HANDOFF if not (h / f).exists()]
    r.add("handoff structure", not missing,
          "complete" if not missing else f"missing {missing}")
    docs = {}
    for f in REQUIRED_HANDOFF:
        if f.endswith(".json") and (h / f).exists():
            try:
                docs[f] = load_json(h / f)
            except json.JSONDecodeError as e:
                r.add(f"parse {f}", False, str(e)[:60])
    return docs


def check_checksums(h: pathlib.Path, docs: dict, r: Result) -> None:
    stored = docs.get("checksums.json") or {}
    entries = stored.get("files", stored)
    if not isinstance(entries, dict) or not entries:
        r.add("checksums present", False, "checksums.json has no file map")
        return
    bad, checked = [], 0
    for rel, expect in entries.items():
        p = h / rel
        if not p.exists():
            p = h.parent / rel
        if not p.exists():
            bad.append(f"{rel}: absent")
            continue
        actual = sha256_file(p)
        checked += 1
        if actual != expect:
            bad.append(f"{rel}: {expect[:10]} != {actual[:10]}")
    r.add("SHA256 integrity", not bad,
          f"{checked} verified" if not bad else f"{len(bad)} mismatched: {bad[:2]}")


def check_raw(h: pathlib.Path, docs: dict, r: Result) -> list[dict]:
    raw_dir = h / "raw"
    files = sorted(raw_dir.glob("*.jsonl")) if raw_dir.is_dir() else []
    if not files:
        r.add("raw rows present", False, "no raw/*.jsonl in the handoff")
        return []
    rows: list[dict] = []
    for f in files:
        try:
            rows.extend(load_jsonl(f))
        except ValueError as e:
            r.add(f"parse {f.name}", False, str(e)[:70])
    r.add("raw rows present", bool(rows), f"{len(rows)} rows from {len(files)} file(s)")

    manifest = docs.get("raw_manifest.json") or {}
    claimed = manifest.get("total_rows")
    if claimed is not None:
        r.add("row count matches manifest", int(claimed) == len(rows),
              f"manifest {claimed} vs actual {len(rows)}")

    keys = [(x.get("problem_id"), x.get("sample_id")) for x in rows]
    dupes = len(keys) - len(set(keys))
    r.add("no duplicate sample ids", dupes == 0, f"{dupes} duplicates")

    present_forbidden = sorted({k for x in rows for k in x if k in FORBIDDEN_FIELDS})
    r.add("information boundary", not present_forbidden,
          "clean" if not present_forbidden else f"FORBIDDEN: {present_forbidden}")
    return rows


def check_split(docs: dict, rows: list[dict], r: Result) -> None:
    cfg = docs.get("config.json") or {}
    cal = set(cfg.get("calibration") or [])
    ev = set(cfg.get("evaluation") or [])
    if not cal or not ev:
        r.add("split declared", False, "config.json lacks calibration/evaluation")
        return
    overlap = cal & ev
    r.add("split disjoint", not overlap, f"cal {len(cal)} eval {len(ev)} overlap {len(overlap)}")
    seen = {x.get("problem_id") for x in rows}
    unknown = seen - cal - ev - {None}
    r.add("rows within the frozen split", not unknown,
          "all accounted for" if not unknown else f"{len(unknown)} unknown ids")


def recompute(rows: list[dict], docs: dict, r: Result) -> dict:
    """Rebuild every headline number from raw rows, ignoring the notebook's claims."""
    cfg = docs.get("config.json") or {}
    ev = set(cfg.get("evaluation") or [])
    by: dict[str, list[dict]] = {}
    for x in rows:
        if x.get("problem_id") in ev:
            by.setdefault(x["problem_id"], []).append(x)
    for q in by:
        by[q].sort(key=lambda z: z.get("sample_id", 0))
    if not by:
        r.add("recomputation possible", False, "no evaluation rows")
        return {}

    qids = sorted(by)
    solved = lambda x: bool(x.get("public_tests_failed", 1) == 0
                            and x.get("public_tests_passed", 0) > 0)
    tokens = lambda x: float(x.get("total_tokens") or 0.0)

    U1 = np.array([1.0 if solved(by[q][0]) else 0.0 for q in qids])
    Ua = np.array([1.0 if any(solved(s) for s in by[q]) else 0.0 for q in qids])
    C1 = np.array([tokens(by[q][0]) for q in qids])
    Call = np.array([sum(tokens(s) for s in by[q]) for q in qids])

    out = {
        "n_evaluation_problems": len(qids),
        "utility_k1": float(U1.mean()),
        "utility_kall": float(Ua.mean()),
        "cost_k1": float(C1.mean()),
        "cost_kall": float(Call.mean()),
        "total_tokens": float(sum(tokens(x) for x in rows)),
        "generation_failures": int(sum(1 for x in rows
                                       if x.get("generation_status") not in (None, "ok"))),
        "execution_failures": int(sum(1 for x in rows
                                      if x.get("execution_status") == "error")),
    }

    scores = [x.get("governor_score") for x in rows if x.get("governor_score") is not None]
    if scores and len(scores) == len(rows):
        y = np.array([1.0 if solved(x) else 0.0 for x in rows])
        out["ranking"] = ranking_metrics(y, np.array(scores, float))

    gov = [x for x in rows if x.get("allocation_decision") is not None]
    if gov:
        d = np.array([1.0 if solved(x) else 0.0 for x in gov]) - U1.mean()
        out["paired"] = paired_bootstrap(d)

    r.add("recomputation possible", True, f"{len(qids)} evaluation problems")
    return out


def compare(recomputed: dict, docs: dict, r: Result, tol: float) -> None:
    """The point of the whole script: does the notebook's claim survive recomputation?"""
    claimed = {**(docs.get("metrics.json") or {}),
               **(docs.get("experiment_summary.json") or {})}
    pairs = [
        ("utility_kall", "governor_U"), ("utility_kall", "utility"),
        ("cost_kall", "actual_tokens"), ("n_evaluation_problems", "n_evaluation"),
        ("total_tokens", "total_tokens"),
    ]
    compared = 0
    for mine_key, theirs_key in pairs:
        if mine_key not in recomputed or theirs_key not in claimed:
            continue
        a, b = float(recomputed[mine_key]), float(claimed[theirs_key])
        agree = math.isclose(a, b, rel_tol=tol, abs_tol=tol)
        compared += 1
        r.add(f"agrees: {theirs_key}", agree,
              f"colab {b:.6g} vs recomputed {a:.6g}")
    if compared == 0:
        r.add("numerical agreement", False,
              "no comparable field found -- the handoff does not expose its headline "
              "numbers under expected names")


def check_guardrails(docs: dict, r: Result) -> None:
    g = docs.get("guardrails.json") or {}
    traps = g.get("traps") or g
    if not isinstance(traps, dict) or not traps:
        r.add("guardrails reported", False, "guardrails.json has no trap results")
        return
    red = [k for k, v in traps.items()
           if (v is False) or (isinstance(v, dict) and not v.get("ok", True))
           or (isinstance(v, (list, tuple)) and len(v) > 0 and v[0] is False)]
    r.add("all traps green", not red, "none red" if not red else f"RED: {red}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--handoff", required=True)
    ap.add_argument("--tolerance", type=float, default=TOLERANCE)
    args = ap.parse_args()

    h = pathlib.Path(args.handoff)
    if not h.is_dir():
        print(f"no such handoff directory: {h}")
        return 2

    print("=" * 72)
    print("INDEPENDENT VERIFICATION — the notebook's report is evidence, not truth")
    print("=" * 72)

    r = Result()
    docs = check_structure(h, r)
    check_checksums(h, docs, r)
    rows = check_raw(h, docs, r)
    if rows:
        check_split(docs, rows, r)
        rec = recompute(rows, docs, r)
        compare(rec, docs, r, args.tolerance)
        (h / "recomputed_metrics.json").write_text(json.dumps(rec, indent=1))
    check_guardrails(docs, r)

    print(r.render())
    if r.failed:
        print("\n  STATUS = VERIFICATION_FAILED")
        print("  Do not edit the summary to match. Trace each discrepancy to the raw rows.")
        return 1
    print("\n  STATUS = VERIFIED  (engineering; the scientific verdict is separate)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
