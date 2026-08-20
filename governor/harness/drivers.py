"""DRIVER_REPRODUCIBILITY — the check that `make verify` structurally cannot make.

`verify_experiment` re-derives an experiment's reported metric from its own
stored raw rows. That proves the ARITHMETIC is honest. It says nothing about
whether the code that produced those rows still exists.

The distinction was not theoretical. A repository cleanup deleted 34 experiment
driver scripts -- the means of regenerating E0012, E0016-E0018, E0020, E0022,
E0023-E0025 and the env6 reference -- and every check stayed green. `make verify`
passed, 272 tests passed, the ledger verified 26/26. The reachability analysis
that authorised the deletion looked for scripts CITED BY NAME in configs, docs
and tests, and nothing cites a driver that way: an experiment records its commit,
config and metrics, not the filename that produced it.

So a repository whose central claim is reproducibility had quietly lost it, with
no signal anywhere.

Two kinds of reproducibility, tracked separately from now on:

    ARTIFACT   the stored rows reproduce the reported metric
               -- `verify_experiment`, already enforced

    DRIVER     the code that generated those rows is still present
               -- this module

An experiment with no surviving driver is not a failure; some predate the
convention. But it must say so EXPLICITLY, as ARTIFACT_ONLY with a reason, so
that the absence is a recorded decision rather than an accident nobody noticed.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = ROOT / "experiments"
REGISTRY = EXPERIMENTS / "DRIVERS.json"

DRIVER = "DRIVER"
ARTIFACT_ONLY = "ARTIFACT_ONLY"


def load_registry() -> dict:
    if not REGISTRY.exists():
        return {}
    return json.loads(REGISTRY.read_text())


def finalized_experiments() -> list[str]:
    return sorted(p.parent.name for p in EXPERIMENTS.glob("*/results.json"))


def driver_status() -> dict[str, dict]:
    """Per experiment: what it claims, and whether that claim holds."""
    reg = load_registry()
    out: dict[str, dict] = {}
    for exp in finalized_experiments():
        entry = reg.get(exp)
        if entry is None:
            out[exp] = {"status": "UNREGISTERED", "ok": False,
                        "detail": "not in DRIVERS.json -- declare a driver or ARTIFACT_ONLY"}
            continue
        status = entry.get("status")
        if status == ARTIFACT_ONLY:
            reason = entry.get("reason", "")
            out[exp] = {"status": ARTIFACT_ONLY, "ok": bool(reason),
                        "detail": reason or "ARTIFACT_ONLY requires a reason"}
            continue
        driver = entry.get("driver", "")
        present = bool(driver) and (ROOT / driver).exists()
        out[exp] = {"status": DRIVER, "ok": present, "driver": driver,
                    "detail": driver if present else f"MISSING: {driver}"}
    return out


def summary() -> dict:
    st = driver_status()
    return {
        "total": len(st),
        "with_driver": sum(1 for v in st.values() if v["status"] == DRIVER and v["ok"]),
        "artifact_only": sum(1 for v in st.values() if v["status"] == ARTIFACT_ONLY and v["ok"]),
        "broken": sorted(k for k, v in st.items() if not v["ok"]),
    }


def render() -> str:
    st = driver_status()
    lines = []
    for exp, v in st.items():
        mark = "  ok  " if v["ok"] else " FAIL "
        lines.append(f"  [{mark}] {exp:<30} {v['status']:<14} {v['detail'][:60]}")
    s = summary()
    lines.append(
        f"\n  {s['with_driver']} with a driver, {s['artifact_only']} artifact-only, "
        f"{len(s['broken'])} broken")
    if s["broken"]:
        lines.append(f"  BROKEN: {', '.join(s['broken'])}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(render())
