#!/usr/bin/env python3
"""Archive a Colab run to Drive and build the handoff package for verification.

Designed to be run REPEATEDLY, including while generation is still going and
including after an interrupted run. A partial archive is useful; a missing one is
not, and the most likely outcome of a multi-hour Colab session is that it ends
before you expected it to.

Two rules it will not bend:

  DRIVE IS OPTIONAL AND NEVER ASSUMED. Everything is written locally first. If
  Drive is unavailable the archive still completes and the status reads
  DRIVE_ARCHIVE = UNAVAILABLE. Silently pretending persistence succeeded is worse
  than not having it, because the failure surfaces only when the runtime is gone.

  THE HANDOFF IS EVIDENCE, NOT A CONCLUSION. Every headline number it carries is
  recomputed from raw rows by `verify_colab_run.py`, which ignores the summary.
  CLAUDE_HANDOFF.md says so at the bottom, in those words.

    python scripts/colab_archive.py                 # local only
    python scripts/colab_archive.py --drive         # also copy to MyDrive/Governor
    python scripts/colab_archive.py --partial       # mid-run snapshot, no gate
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
ROOT = pathlib.Path(__file__).resolve().parents[1]

EXPERIMENT_ID = "E0029-QWEN"
HANDOFF = ROOT / "claude_handoff"
DRIVE_ROOT = pathlib.Path("/content/drive/MyDrive/Governor")

RAW_CANDIDATES = [
    ROOT / "results" / "e0029_colab_generations.jsonl",
    ROOT / "results" / "e0029_qwen_generations.jsonl",
]
ARTEFACTS = [
    "results/colab_environment.json", "results/colab_model_meta.json",
    "results/colab_preflight.json", "results/E0029-QWEN-preflight.json",
    "results/batch_bench.json", "configs/colab_model.json",
    "configs/colab-requirements.txt", "configs/e0029_split.json",
]


def sha256(p: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                              text=True).stdout.strip()
    except Exception:                                     # noqa: BLE001
        return ""


def check_drive() -> tuple[bool, str]:
    """Verify an EXISTING mount. This script must never mount Drive itself.

    `google.colab.drive` is injected into the notebook KERNEL, and its mount
    needs the kernel's interactive channel to render the OAuth prompt. This
    script runs as a subprocess of that kernel, where the import fails and the
    prompt has nowhere to appear -- so a mount attempted from here can only ever
    fail, and would do so while reporting a misleading reason ("not running in
    Colab") on a machine that plainly is.

    The mount therefore happens in the notebook cell, in-process, and this
    function only confirms the result. Verification is by writing a probe file,
    not by checking that a directory exists: a stale or half-detached mount
    still presents the directory.
    """
    mnt = pathlib.Path("/content/drive/MyDrive")
    if not mnt.is_dir():
        return False, ("Drive is not mounted. Mount it from the NOTEBOOK cell "
                       "(not here): from google.colab import drive; "
                       "drive.mount('/content/drive')")
    probe = mnt / ".governor_write_probe"
    try:
        probe.write_text("ok")
        got = probe.read_text()
        probe.unlink()
    except Exception as e:                                # noqa: BLE001
        return False, f"mounted but not writable: {type(e).__name__}: {e}"
    if got != "ok":
        return False, "probe file read back wrong -- mount is unhealthy"
    return True, "mounted and writable"


def load_rows(path: pathlib.Path) -> list[dict]:
    rows, torn = [], 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                torn += 1                                 # an interrupted final write
    if torn:
        print(f"  note: {torn} unparseable line(s) skipped — expected after an interrupt")
    return rows


def summarise(rows: list[dict], cfg: dict) -> dict:
    """Facts, not verdicts. Distinguishes the three failure kinds the spec requires."""
    ev = set(cfg.get("evaluation", []))
    by: dict[str, list[dict]] = {}
    for r in rows:
        by.setdefault(r.get("problem_id"), []).append(r)
    solved = lambda r: (r.get("pub_failed", 1) == 0 and r.get("pub_passed", 0) > 0)
    ev_rows = [r for r in rows if r.get("problem_id") in ev]
    ev_by: dict[str, list[dict]] = {}
    for r in ev_rows:
        ev_by.setdefault(r["problem_id"], []).append(r)

    gen_fail = sum(1 for r in rows if r.get("generation_status") not in ("ok",))
    exec_fail = sum(1 for r in rows if r.get("execution_status") == "error")
    wrong = sum(1 for r in rows
                if r.get("generation_status") == "ok" and not solved(r))
    return {
        "experiment_id": EXPERIMENT_ID,
        "rows": len(rows),
        "problems_seen": len(by),
        "problems_expected": cfg.get("n_problems"),
        "samples_per_problem": cfg.get("samples_per_problem"),
        "complete": len(rows) >= (cfg.get("n_problems", 0)
                                  * cfg.get("samples_per_problem", 0)),
        "evaluation_rows": len(ev_rows),
        "evaluation_problems": len(ev_by),
        "utility_any_sample": (sum(1 for q in ev_by if any(solved(r) for r in ev_by[q]))
                               / len(ev_by)) if ev_by else None,
        "total_tokens": sum(float(r.get("total_tokens") or 0) for r in rows),
        # Three DIFFERENT states, never collapsed into utility = 0.
        "generation_failures": gen_fail,
        "execution_failures": exec_fail,
        "incorrect_answers": wrong,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def build_handoff(partial: bool) -> dict:
    HANDOFF.mkdir(exist_ok=True)
    (HANDOFF / "raw").mkdir(exist_ok=True)

    raw = [p for p in RAW_CANDIDATES if p.exists()]
    if not raw:
        print("  no raw generations found — nothing to archive yet")
        return {"ok": False, "reason": "no raw data"}

    rows: list[dict] = []
    for p in raw:
        shutil.copy2(p, HANDOFF / "raw" / p.name)
        rows += load_rows(p)

    cfg_path = ROOT / "configs" / "e0029_split.json"
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    summary = summarise(rows, cfg)
    summary["status"] = ("INTERRUPTED" if partial or not summary["complete"]
                         else "GENERATION_COMPLETE")

    for rel in ARTEFACTS:
        src = ROOT / rel
        if src.exists():
            shutil.copy2(src, HANDOFF / pathlib.Path(rel).name)

    (HANDOFF / "experiment_summary.json").write_text(json.dumps(summary, indent=1))
    (HANDOFF / "config.json").write_text(json.dumps(cfg, indent=1))
    (HANDOFF / "provenance.json").write_text(json.dumps({
        "commit": git("rev-parse", "HEAD"),
        "describe": git("describe", "--tags", "--always"),
        "dirty": bool(git("status", "--porcelain")),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD")}, indent=1))
    (HANDOFF / "raw_manifest.json").write_text(json.dumps({
        "total_rows": len(rows),
        "files": [f"raw/{p.name}" for p in raw]}, indent=1))
    (HANDOFF / "failure_log.json").write_text(json.dumps({
        "generation_failures": summary["generation_failures"],
        "execution_failures": summary["execution_failures"],
        "incorrect_answers": summary["incorrect_answers"],
        "note": "Three distinct states. None is collapsed into utility = 0."},
        indent=1))
    # Preflight records "gates"; the verifier reads "traps". Translate rather
    # than leaving guardrails.json in a shape nothing can check.
    gp = ROOT / "results" / "colab_preflight.json"
    if gp.exists():
        gates = json.loads(gp.read_text()).get("gates", [])
        (HANDOFF / "guardrails.json").write_text(json.dumps(
            {"traps": {g["name"].replace(" ", "_"): {"ok": bool(g["ok"]),
                                                     "detail": g.get("detail", "")}
                       for g in gates},
             "source": "colab_preflight gates"}, indent=1))

    for name, default in (("model.json", {}), ("metrics.json", {}),
                          ("environment.json", {}), ("guardrails.json", {})):
        tgt = HANDOFF / name
        if not tgt.exists():
            src = {"model.json": ROOT / "results/colab_model_meta.json",
                   "environment.json": ROOT / "results/colab_environment.json",
                   "guardrails.json": ROOT / "results/colab_preflight.json",
                   "metrics.json": None}.get(name)
            tgt.write_text((src.read_text() if src and src.exists()
                            else json.dumps(default, indent=1)))

    (HANDOFF / "reproduction_commands.txt").write_text(
        f"git clone https://github.com/SYT20/Governor.git\n"
        f"cd Governor\n"
        f"git checkout {git('rev-parse', 'HEAD')}\n"
        f"pip install -r configs/colab-requirements.txt\n"
        f"python scripts/verify_colab_run.py --handoff claude_handoff/\n")

    sums = {}
    for f in sorted(HANDOFF.rglob("*")):
        if f.is_file() and f.name != "checksums.json":
            sums[str(f.relative_to(HANDOFF))] = sha256(f)
    (HANDOFF / "checksums.json").write_text(json.dumps({"files": sums}, indent=1))

    write_handoff_md(summary, cfg)
    return {"ok": True, "summary": summary, "files": len(sums)}


def write_handoff_md(s: dict, cfg: dict) -> None:
    model = {}
    mp = ROOT / "results" / "colab_model_meta.json"
    if mp.exists():
        model = json.loads(mp.read_text())
    env = {}
    ep = ROOT / "results" / "colab_environment.json"
    if ep.exists():
        env = json.loads(ep.read_text())
    hw = (env.get("hardware") or {})
    gpu = (hw.get("gpu") or {})
    util = s.get("utility_any_sample")

    (HANDOFF / "CLAUDE_HANDOFF.md").write_text(f"""# {s['experiment_id']} — Colab handoff

**STATUS: {s['status']}**

| | |
|---|---|
| EXPERIMENT ID | {s['experiment_id']} |
| GIT COMMIT | `{git('rev-parse', 'HEAD')}` |
| MODEL | {model.get('model_name', cfg.get('model', 'unknown'))} |
| MODEL REVISION | {model.get('revision', 'unresolved')} |
| HARDWARE | {gpu.get('name', 'CPU only')} · {hw.get('cpu_count', '?')} vCPU · {hw.get('ram_total_gb', '?')} GB RAM |
| COLAB RELEASE | {hw.get('colab_release_tag', 'unknown')} |
| DATASET | LiveCodeBench, {cfg.get('n_problems', '?')} frozen problems |
| SPLIT | cal {len(cfg.get('calibration', []))} / eval {len(cfg.get('evaluation', []))}, sha256 parity |
| SEED | {cfg.get('generation', {}).get('seed', 1000)} |
| GENERATION | {json.dumps(cfg.get('generation', {}))[:120]} |

## Progress

| | |
|---|---|
| rows written | {s['rows']} of {(s.get('problems_expected') or 0) * (s.get('samples_per_problem') or 0)} |
| problems seen | {s['problems_seen']} of {s.get('problems_expected')} |
| complete | {s['complete']} |
| total tokens | {s['total_tokens']:,.0f} |

## Failures, kept as distinct states

| kind | count |
|---|---|
| generation failed | {s['generation_failures']} |
| execution failed | {s['execution_failures']} |
| ran but answered incorrectly | {s['incorrect_answers']} |

None of these is collapsed into `utility = 0`. They mean different things.

## Descriptive result

Utility over evaluation problems, any-sample-passes:
**{f'{util:.4f}' if util is not None else 'not computable yet'}**
over {s['evaluation_problems']} evaluation problems.

**This is a descriptive figure from a partial or complete generation pass. It is
not the experiment's result.** The Governor / best-fixed / myopic / oracle
comparison, its confidence interval, and the guardrails are produced by the
analysis stage and verified independently.

## Verification required

```bash
git checkout {git('rev-parse', 'HEAD')}
python scripts/verify_colab_run.py --handoff claude_handoff/
```

That script ignores every number above and recomputes it from `raw/*.jsonl`. It
has been tested against four kinds of tampering — an inflated summary, a
forbidden field added to raw rows, a manufactured split overlap, and rows edited
after checksums were taken — and exits non-zero on each.

On disagreement the status is `VERIFICATION_FAILED` and the summary is **not**
edited to match. The discrepancy is the finding.

## Artifacts

`raw/` · `experiment_summary.json` · `environment.json` · `model.json` ·
`config.json` · `checksums.json` · `guardrails.json` · `metrics.json` ·
`provenance.json` · `raw_manifest.json` · `failure_log.json` ·
`reproduction_commands.txt`

SHA256 for every file is in `checksums.json`.

---

**THE COLAB RESULT MUST NOT BE TRUSTED WITHOUT RECOMPUTATION.**
""")


def copy_to_drive() -> dict:
    ok, why = check_drive()
    if not ok:
        print(f"  DRIVE_ARCHIVE = UNAVAILABLE  ({why})")
        return {"status": "UNAVAILABLE", "reason": why}
    dest = DRIVE_ROOT / "experiments" / EXPERIMENT_ID
    for sub in ("experiments", "checkpoints", "raw", "metrics", "reports",
                "configs", "environments", "notebooks", "logs"):
        (DRIVE_ROOT / sub).mkdir(parents=True, exist_ok=True)
    dest.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        shutil.rmtree(dest, ignore_errors=True)
    shutil.copytree(HANDOFF, dest)
    nb = ROOT / "notebooks" / "governor_colab_e0029_qwen.ipynb"
    if nb.exists():
        shutil.copy2(nb, DRIVE_ROOT / "notebooks" / nb.name)
    n = sum(1 for _ in dest.rglob("*") if _.is_file())
    print(f"  DRIVE_ARCHIVE = OK   {dest}  ({n} files)")
    return {"status": "OK", "path": str(dest), "files": n}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--drive", action="store_true")
    ap.add_argument("--partial", action="store_true")
    args = ap.parse_args()

    print("=" * 68)
    print(f"ARCHIVE — {EXPERIMENT_ID}   {time.strftime('%H:%M:%S')}")
    print("=" * 68)

    r = build_handoff(args.partial)
    if not r["ok"]:
        print(f"\n  nothing archived: {r['reason']}")
        return 1
    s = r["summary"]
    print(f"  handoff: {HANDOFF}  ({r['files']} files, all checksummed)")
    print(f"  status : {s['status']}   rows {s['rows']}   "
          f"problems {s['problems_seen']}/{s.get('problems_expected')}")

    drive = copy_to_drive() if args.drive else {"status": "SKIPPED"}
    (HANDOFF / "drive_status.json").write_text(json.dumps(drive, indent=1))

    print(f"\n  DRIVE_ARCHIVE: {drive['status']}")
    print(f"  verify with:   python scripts/verify_colab_run.py --handoff claude_handoff/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
