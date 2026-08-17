"""Experiment ledger. Provenance is a GATE, not a log.

WHY THIS EXISTS. This project produced the number U=0.8247 and then spent a
session unable to prove it. The mean survived in a markdown file; the 800
per-episode outcomes that produced it did not. A summary without raw data is a
claim, not a result, and three of this project's retractions were only possible
to diagnose because raw rows happened to still be on disk.

So `record_experiment` REFUSES to finalize unless everything needed to
reconstruct the number exists and agrees:

    git commit hash        which code produced this
    model identifier       which engine answered
    runtime/version        which interpreter and library versions
    budget                 the resource cap that was enforced
    seed(s)                the draw
    split                  train / calibration / test, stated not implied
    metric definition      what the number means, in words
    raw result file        one line per episode, hashed
    summary result         the headline
    config file            everything above, machine-readable

and then verifies, by execution:

    the raw file exists and is non-empty
    every raw row carries THIS run's nonce (no stale file reuse)
    the raw file's mtime is inside this run's window
    the recorded config hash matches the config on disk
    HEAD still matches git_commit.txt (no edits mid-run)
    the tree is clean (a dirty tree makes the commit hash a lie)

A finalized experiment also stores the trap-check verdict. A red trap forces
`verdict="BLOCKED"` no matter what the deltas say -- the caller does not get to
decide that, because in every historical case the caller was me and I had
already printed PASS.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = ROOT / "experiments"

REQUIRED_CONFIG_FIELDS = (
    "exp_id", "title", "model", "runtime", "budget", "seeds", "split",
    "metric", "git_commit",
)


class ProvenanceError(RuntimeError):
    """Raised instead of writing an unreproducible result."""


# -- git ---------------------------------------------------------------------

def _git(*args: str) -> str:
    try:
        return subprocess.run(("git", *args), cwd=ROOT, capture_output=True,
                              text=True, timeout=30).stdout.strip()
    except Exception:                                        # noqa: BLE001
        return ""


def git_commit() -> str:
    return _git("rev-parse", "HEAD")


def git_dirty(exclude: Path | None = None) -> list[str]:
    """Paths that differ from HEAD, ignoring anything under `exclude`.

    An experiment writes its own directory while it runs, so counting those
    files as drift would make every experiment unfinalizable. What must be clean
    is the CODE -- everything outside the run's own output.
    """
    out = []
    rel = None
    if exclude is not None:
        try:
            rel = str(exclude.resolve().relative_to(ROOT))
        except ValueError:
            rel = None
    for line in _git("status", "--porcelain").splitlines():
        # Slicing [3:] loses a character: `_git` strips the output, so the
        # leading space of a " M path" status column is already gone on the
        # first line. Split on whitespace instead of counting columns.
        parts = line.strip().split(maxsplit=1)
        if len(parts) < 2:
            continue
        path = parts[1].strip('"').split(" -> ")[-1]
        if rel and (path == rel or path.startswith(rel + "/")):
            continue
        out.append(path)
    return out


def file_commit(path: str) -> str:
    """Commit that last changed `path`.

    Used as the `frozen_before_heldout` evidence: the preregistration's commit
    must predate the commit that produced the held-out numbers. Comparing the
    run's commit to itself would make that check meaningless, which is what it
    was in its first form.
    """
    return _git("log", "-1", "--format=%H", "--", path)


def runtime_fingerprint() -> dict[str, str]:
    """Versions of everything that can move a number without a commit."""
    out = {"python": sys.version.split()[0], "platform": platform.platform(),
           "machine": platform.machine()}
    for mod in ("numpy", "sklearn", "scipy"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:                                    # noqa: BLE001
            out[mod] = "absent"
    return out


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _canonical_hash(obj: Any) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


# -- spec --------------------------------------------------------------------

@dataclass(slots=True)
class ExperimentSpec:
    """Everything that must be known BEFORE the first episode runs.

    Declaring the split and the metric up front is the point: both of this
    project's selection-leakage retractions came from choosing them after
    seeing results.
    """
    exp_id: str
    title: str
    model: str                       # engine identifier, e.g. the model slug
    budget: dict[str, Any]           # what was capped, in what units
    seeds: dict[str, int]            # named seeds: calibration, test, ...
    split: dict[str, Any]            # sizes and provenance of each split
    metric: str                      # the metric DEFINITION, in words
    params: dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    def to_config(self, exclude_dir: Path | None = None) -> dict[str, Any]:
        d = asdict(self)
        d["runtime"] = runtime_fingerprint()
        d["git_commit"] = git_commit()
        d["git_dirty"] = git_dirty(exclude=exclude_dir)
        return d


# -- run ---------------------------------------------------------------------

class ExperimentRun:
    """Open an experiment, stream raw rows, then finalize.

    Raw rows are written as they are produced, so a run killed halfway leaves
    evidence of exactly how far it got instead of nothing. `results.json` is
    written only by `finalize`, so an unfinalized directory is visibly partial.
    """

    def __init__(self, spec: ExperimentSpec, root: Path | None = None,
                 overwrite: bool = False):
        self.spec = spec
        self.dir = (root or EXPERIMENTS) / spec.exp_id
        if self.dir.exists() and not overwrite and (self.dir / "results.json").exists():
            raise ProvenanceError(
                f"{spec.exp_id} already finalized. Use a new exp_id -- "
                f"overwriting a finalized experiment destroys the audit trail.")
        self.dir.mkdir(parents=True, exist_ok=True)
        self.nonce = uuid.uuid4().hex          # stamps every raw row
        self.t0 = time.time()
        self.commit = git_commit()
        self.config = self.spec.to_config(exclude_dir=self.dir)
        self.config["nonce"] = self.nonce
        self.config["started_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                   time.gmtime(self.t0))
        self.config_hash = _canonical_hash(self.config)
        (self.dir / "config.json").write_text(json.dumps(self.config, indent=2,
                                                         default=str))
        (self.dir / "git_commit.txt").write_text(self.commit + "\n")
        self.raw_path = self.dir / "raw.jsonl"
        self.raw_path.write_text("")           # fresh; never append to a stale file
        self._raw = self.raw_path.open("a")
        self.n_rows = 0

    def append(self, row: dict[str, Any]) -> None:
        """One raw row. The nonce is what proves the row belongs to this run."""
        row = {**row, "_nonce": self.nonce}
        self._raw.write(json.dumps(row, default=str) + "\n")
        self.n_rows += 1

    def flush(self) -> None:
        self._raw.flush()
        os.fsync(self._raw.fileno())

    # -- verification --------------------------------------------------------

    def _verify(self) -> list[str]:
        """Return the list of provenance failures. Empty list == finalizable."""
        bad: list[str] = []

        for f in REQUIRED_CONFIG_FIELDS:
            v = self.config.get(f)
            if v in (None, "", {}, []):
                bad.append(f"config missing required field: {f}")

        if not self.raw_path.exists():
            bad.append("raw file does not exist")
            return bad
        if self.n_rows == 0 or self.raw_path.stat().st_size == 0:
            bad.append("raw file is empty -- summary without raw data is refused")
            return bad

        mtime = self.raw_path.stat().st_mtime
        if not (self.t0 - 1.0 <= mtime <= time.time() + 1.0):
            bad.append(f"raw file mtime {mtime} outside this run's window "
                       f"-- it is a leftover from another run")

        n_seen, n_wrong = 0, 0
        for line in self.raw_path.read_text().splitlines():
            if not line.strip():
                continue
            n_seen += 1
            try:
                if json.loads(line).get("_nonce") != self.nonce:
                    n_wrong += 1
            except json.JSONDecodeError:
                n_wrong += 1
        if n_wrong:
            bad.append(f"{n_wrong}/{n_seen} raw rows do not carry this run's nonce")
        if n_seen != self.n_rows:
            bad.append(f"raw row count {n_seen} != rows written {self.n_rows}")

        on_disk = json.loads((self.dir / "config.json").read_text())
        if _canonical_hash(on_disk) != self.config_hash:
            bad.append("config.json on disk differs from the recorded config")

        head = git_commit()
        recorded = (self.dir / "git_commit.txt").read_text().strip()
        if head != recorded:
            bad.append(f"HEAD moved during the run: {recorded[:8]} -> {head[:8]}")
        dirty = git_dirty(exclude=self.dir)
        if dirty:
            bad.append(f"working tree is dirty -- the commit hash does not "
                       f"describe the code that ran: {dirty[:6]}")
        return bad

    # -- finalize ------------------------------------------------------------

    def finalize(self, summary: dict[str, Any], metrics: dict[str, Any],
                 traps: dict[str, tuple[bool, str]] | None = None,
                 verdict: str = "UNSET", readme: str = "",
                 allow_dirty: bool = False) -> dict[str, Any]:
        """Write results.json + metrics.json, or refuse and explain."""
        self.flush()
        if not summary:
            raise ProvenanceError("refusing to finalize with an empty summary")
        problems = self._verify()
        if allow_dirty:
            problems = [p for p in problems if "dirty" not in p]
        if problems:
            raise ProvenanceError(
                f"{self.spec.exp_id} NOT finalized. Provenance failures:\n  - "
                + "\n  - ".join(problems))

        traps = traps or {}
        red = sorted(n for n, (ok, _) in traps.items() if not ok)
        if red:
            verdict = "BLOCKED"          # not the caller's call

        results = {
            "exp_id": self.spec.exp_id,
            "title": self.spec.title,
            "verdict": verdict,
            "summary": summary,
            "git_commit": self.commit,
            "config_hash": self.config_hash,
            "nonce": self.nonce,
            "raw_file": self.raw_path.name,
            "raw_sha256": sha256_file(self.raw_path),
            "raw_rows": self.n_rows,
            "wall_s": round(time.time() - self.t0, 1),
            "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "trap_checks": {n: {"ok": ok, "detail": d} for n, (ok, d) in traps.items()},
            "red_traps": red,
        }
        (self.dir / "results.json").write_text(json.dumps(results, indent=2,
                                                          default=str))
        (self.dir / "metrics.json").write_text(json.dumps(metrics, indent=2,
                                                          default=str))
        (self.dir / "README.md").write_text(
            readme or _default_readme(self.spec, results, metrics))
        self._raw.close()
        return results

    def abort(self, reason: str) -> None:
        """Kill a run without leaving a finalized-looking directory behind."""
        self.flush()
        self._raw.close()
        (self.dir / "ABORTED.txt").write_text(
            f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n{reason}\n")


def _default_readme(spec: ExperimentSpec, results: dict, metrics: dict) -> str:
    trap_lines = "\n".join(
        f"- {'GREEN' if v['ok'] else '**RED**'} `{k}` — {v['detail']}"
        for k, v in results["trap_checks"].items()) or "- (none supplied)"
    return f"""# {spec.exp_id} — {spec.title}

**Verdict: {results['verdict']}**

| field | value |
|---|---|
| model | `{spec.model}` |
| commit | `{results['git_commit']}` |
| budget | `{json.dumps(spec.budget)}` |
| seeds | `{json.dumps(spec.seeds)}` |
| split | `{json.dumps(spec.split)}` |
| raw rows | {results['raw_rows']} (`{results['raw_file']}`, sha256 `{results['raw_sha256'][:16]}`) |
| wall | {results['wall_s']} s |

## Metric

{spec.metric}

## Summary

```json
{json.dumps(results['summary'], indent=2, default=str)}
```

## Trap checks

{trap_lines}

## Notes

{spec.notes}
"""


# -- reading -----------------------------------------------------------------

def load_experiment(exp_id: str, root: Path | None = None) -> dict[str, Any]:
    d = (root or EXPERIMENTS) / exp_id
    res = json.loads((d / "results.json").read_text())
    res["config"] = json.loads((d / "config.json").read_text())
    res["metrics"] = json.loads((d / "metrics.json").read_text())
    return res


def verify_experiment(exp_id: str, root: Path | None = None) -> tuple[bool, list[str]]:
    """Re-verify a FINALIZED experiment from disk, later, by anyone.

    This is the check that makes the ledger worth having: it can be run months
    later against a directory nobody remembers producing.
    """
    d = (root or EXPERIMENTS) / exp_id
    bad: list[str] = []
    for f in ("config.json", "results.json", "metrics.json", "raw.jsonl",
              "git_commit.txt", "README.md"):
        if not (d / f).exists():
            bad.append(f"missing {f}")
    if bad:
        return False, bad
    res = json.loads((d / "results.json").read_text())
    cfg = json.loads((d / "config.json").read_text())
    raw = d / "raw.jsonl"

    if sha256_file(raw) != res["raw_sha256"]:
        bad.append("raw.jsonl has been modified since finalization")
    rows = [line for line in raw.read_text().splitlines() if line.strip()]
    if len(rows) != res["raw_rows"]:
        bad.append(f"raw row count {len(rows)} != recorded {res['raw_rows']}")
    if any(json.loads(r).get("_nonce") != res["nonce"] for r in rows):
        bad.append("raw rows carry a foreign nonce")
    if _canonical_hash(cfg) != res["config_hash"]:
        bad.append("config.json does not match the recorded config hash")
    if cfg.get("git_commit") != res["git_commit"]:
        bad.append("config commit != results commit")
    if (d / "git_commit.txt").read_text().strip() != res["git_commit"]:
        bad.append("git_commit.txt != results commit")
    if res["red_traps"] and res["verdict"] != "BLOCKED":
        bad.append(f"red traps {res['red_traps']} but verdict {res['verdict']}")
    return (not bad), bad


def index(root: Path | None = None) -> list[dict[str, Any]]:
    """Every finalized experiment, with its verdict and whether it still verifies."""
    r = root or EXPERIMENTS
    out = []
    for d in sorted(p for p in r.glob("E*") if p.is_dir()):
        if not (d / "results.json").exists():
            out.append({"exp_id": d.name, "verdict": "UNFINALIZED",
                        "verifies": False})
            continue
        res = json.loads((d / "results.json").read_text())
        ok, _ = verify_experiment(d.name, r)
        out.append({"exp_id": d.name, "title": res.get("title", ""),
                    "verdict": res["verdict"], "commit": res["git_commit"][:8],
                    "rows": res["raw_rows"], "verifies": ok})
    return out
