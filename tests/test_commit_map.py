"""Provenance must survive the one history rewrite this repository has had.

An API key was committed during Phase 3 and later purged, which changed 99 of
175 commit hashes. Experiment records were deliberately NOT edited to match --
rewriting a recorded result so it agrees with a later convenience is the exact
failure mode this project's provenance rules exist to prevent.

`experiments/COMMIT-MAP.tsv` is therefore load-bearing: it is the only thing
connecting a recorded hash to a commit that still exists.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MAP = ROOT / "experiments" / "COMMIT-MAP.tsv"


def _table() -> dict[str, str]:
    rows = {}
    for line in MAP.read_text().splitlines():
        if line.startswith("#") or line.startswith("old_sha") or "\t" not in line:
            continue
        old, new = line.split("\t", 1)
        rows[old.strip()] = new.strip()
    return rows


def test_map_exists_and_is_documented():
    assert MAP.exists(), "the commit map is the only bridge to pre-rewrite hashes"
    head = MAP.read_text()[:2000]
    assert head.startswith("#"), "the map must explain why it exists"
    assert "rewritten" in head


def test_every_recorded_hash_resolves():
    table = _table()
    unmapped = []
    for f in sorted(ROOT.glob("experiments/*/git_commit.txt")):
        old = f.read_text().strip()
        if old not in table:
            unmapped.append((f.parent.name, old))
    assert not unmapped, f"experiment hashes absent from the map: {unmapped}"


def test_every_mapped_target_is_a_real_commit():
    """A map pointing at commits that do not exist is worse than no map."""
    table = _table()
    targets = {table[Path(f).read_text().strip()]
               for f in ROOT.glob("experiments/*/git_commit.txt")}
    missing = [t for t in sorted(targets)
               if subprocess.run(["git", "cat-file", "-e", f"{t}^{{commit}}"],
                                 cwd=ROOT, capture_output=True).returncode != 0]
    assert not missing, f"map points at non-existent commits: {missing}"


def test_no_live_credential_survives_in_history():
    """The whole point of the rewrite."""
    out = subprocess.run(["git", "log", "--all", "-p"],
                         cwd=ROOT, capture_output=True, text=True).stdout
    import re
    hits = re.findall(r"sk-or-v1-[A-Za-z0-9]{20,}", out)
    assert not hits, f"{len(hits)} credential(s) still reachable in history"


def test_resolver_audit_passes():
    r = subprocess.run(["python", "scripts/resolve_commit.py", "--audit"],
                       cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "Provenance intact" in r.stdout
