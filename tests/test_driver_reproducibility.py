"""Deleting an experiment's driver must fail a test, not pass silently.

A cleanup once removed 34 driver scripts -- the means of regenerating most of
the ledger -- and every check stayed green, because `make verify` re-derives a
metric from stored rows and never asks whether the code that produced them still
exists. These tests close that gap.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from governor.harness.drivers import (
    ARTIFACT_ONLY, DRIVER, REGISTRY, ROOT, driver_status,
    finalized_experiments, load_registry, summary,
)


def test_registry_exists():
    assert REGISTRY.exists(), "experiments/DRIVERS.json is the record of what can be re-run"


def test_every_finalized_experiment_is_registered():
    """A new experiment must declare a driver or explicitly say it has none."""
    reg = load_registry()
    unregistered = [e for e in finalized_experiments() if e not in reg]
    assert not unregistered, (
        f"unregistered experiments: {unregistered}. Add a driver entry, or "
        f"ARTIFACT_ONLY with a reason.")


def test_no_registered_driver_is_missing():
    """The regression that the cleanup would have tripped."""
    broken = {e: v for e, v in driver_status().items()
              if v["status"] == DRIVER and not v["ok"]}
    assert not broken, (
        f"experiments whose driver was deleted: {list(broken)}. Restore the "
        f"script, or reclassify the experiment as ARTIFACT_ONLY with a reason.")


def test_artifact_only_requires_a_stated_reason():
    """ARTIFACT_ONLY must be a decision on the record, not a shrug."""
    bad = [e for e, v in driver_status().items()
           if v["status"] == ARTIFACT_ONLY and not v["ok"]]
    assert not bad, f"ARTIFACT_ONLY without a reason: {bad}"


def test_registry_has_no_entries_for_absent_experiments():
    known = set(finalized_experiments())
    stale = [e for e in load_registry() if e not in known]
    assert not stale, f"registry references experiments that do not exist: {stale}"


def test_registry_entries_are_well_formed():
    for exp, entry in load_registry().items():
        assert entry.get("status") in (DRIVER, ARTIFACT_ONLY), \
            f"{exp}: status must be {DRIVER} or {ARTIFACT_ONLY}"
        if entry["status"] == DRIVER:
            assert entry.get("driver", "").startswith("scripts/"), \
                f"{exp}: driver must be a path under scripts/"


def test_summary_reports_zero_broken():
    s = summary()
    assert s["broken"] == [], f"broken driver references: {s['broken']}"
    assert s["with_driver"] + s["artifact_only"] == s["total"]


def test_artifact_and_driver_reproducibility_are_distinct():
    """The conceptual point, asserted so it cannot quietly collapse back.

    An experiment can verify from its artifacts while its driver is gone. That
    combination is exactly what went undetected, so it must remain expressible
    and visible rather than being conflated.
    """
    st = driver_status()
    artifact_only = [e for e, v in st.items() if v["status"] == ARTIFACT_ONLY]
    for e in artifact_only:
        assert (ROOT / "experiments" / e / "results.json").exists(), \
            f"{e} is ARTIFACT_ONLY but has no artifacts either"
