#!/usr/bin/env python3
"""Resolve a recorded commit hash to its current one, after the history rewrite.

The repository's history was rewritten once, on 2026-08-18, to purge an API key
that had been committed to `scripts/llm_m2_curve.py` during Phase 3. Removing a
secret from a commit changes that commit's hash and every descendant's, so 99 of
175 commits received new ids.

Experiment records were deliberately NOT edited to match. Rewriting a recorded
result so it agrees with a later convenience is precisely what this project's
provenance rules exist to prevent, and `experiments/*/git_commit.txt` still holds
the hash that was true when the experiment ran.

So the mapping lives beside the data instead:

    python scripts/resolve_commit.py 4f5e1ca8              # one hash
    python scripts/resolve_commit.py --experiment E0021-enforced-math
    python scripts/resolve_commit.py --audit               # check all of them

Exit status is non-zero if a hash cannot be resolved, so this is usable in CI.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAP = ROOT / "experiments" / "COMMIT-MAP.tsv"
EXPERIMENTS = ROOT / "experiments"


def load_map() -> dict[str, str]:
    if not MAP.exists():
        sys.exit(f"missing {MAP.relative_to(ROOT)} — cannot resolve hashes")
    out: dict[str, str] = {}
    for line in MAP.read_text().splitlines():
        if line.startswith("#") or line.startswith("old_sha"):
            continue
        if "\t" in line:
            old, new = line.split("\t", 1)
            out[old.strip()] = new.strip()
    return out


def expand(prefix: str, table: dict[str, str]) -> list[str]:
    """Accept an abbreviated hash, the way git does."""
    if prefix in table:
        return [prefix]
    return [k for k in table if k.startswith(prefix)]


def exists_in_repo(sha: str) -> bool:
    r = subprocess.run(["git", "cat-file", "-e", f"{sha}^{{commit}}"],
                       cwd=ROOT, capture_output=True)
    return r.returncode == 0


def describe(sha: str) -> str:
    r = subprocess.run(["git", "log", "-1", "--format=%h %ad %s", "--date=short", sha],
                       cwd=ROOT, capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else "(not in this repository)"


def resolve_one(prefix: str, table: dict[str, str]) -> int:
    hits = expand(prefix, table)
    if not hits:
        print(f"  {prefix}  NOT FOUND in the commit map")
        return 1
    if len(hits) > 1:
        print(f"  {prefix}  ambiguous — matches {len(hits)} commits")
        return 1

    old = hits[0]
    new = table[old]
    if old == new:
        print(f"  {old[:10]}  unchanged by the rewrite")
    else:
        print(f"  {old[:10]}  ->  {new[:10]}")
    print(f"      {describe(new)}")
    return 0 if exists_in_repo(new) else 1


def audit(table: dict[str, str]) -> int:
    """Every experiment's recorded hash must resolve to a commit that exists."""
    bad = 0
    files = sorted(EXPERIMENTS.glob("*/git_commit.txt"))
    print(f"\n  auditing {len(files)} experiment records\n")
    for f in files:
        exp = f.parent.name
        old = f.read_text().strip()
        new = table.get(old)
        if new is None:
            print(f"  UNMAPPED   {exp:<30} {old[:10]}")
            bad += 1
        elif not exists_in_repo(new):
            print(f"  DANGLING   {exp:<30} {old[:10]} -> {new[:10]}")
            bad += 1
        else:
            moved = "moved  " if old != new else "same   "
            print(f"  ok {moved} {exp:<30} {old[:10]} -> {new[:10]}")

    print(f"\n  {len(files) - bad}/{len(files)} resolve to a commit in this repository.")
    if bad:
        print("  A dangling record means the map and the history disagree.\n")
    else:
        print("  Provenance intact across the rewrite.\n")
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("hash", nargs="*", help="one or more (possibly abbreviated) hashes")
    ap.add_argument("--experiment", "-e", help="resolve the hash recorded by this experiment")
    ap.add_argument("--audit", action="store_true", help="check every experiment record")
    args = ap.parse_args()

    table = load_map()

    if args.audit:
        return audit(table)

    if args.experiment:
        f = EXPERIMENTS / args.experiment / "git_commit.txt"
        if not f.exists():
            sys.exit(f"no such experiment record: {f.relative_to(ROOT)}")
        print(f"\n  {args.experiment}")
        return resolve_one(f.read_text().strip(), table)

    if not args.hash:
        ap.print_help()
        return 2

    print()
    return max(resolve_one(h, table) for h in args.hash)


if __name__ == "__main__":
    raise SystemExit(main())
