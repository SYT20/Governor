"""Every decision-time feature must have a declared, admissible provenance.

The name-based trap list caught `graded` only once someone thought to add the
word, and it forbids `difficulty` for a reason that turned out not to apply to
LiveCodeBench. A name cannot distinguish a field published with the problem from
one computed out of the outcomes being predicted. This manifest can, because it
records WHEN each value was fixed.
"""
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from governor.execfeedback.richfeatures import FEATURE_NAMES
from governor.harness.provenance import (
    ADMISSIBLE, CLASSES, HIDDEN_INFORMATION, MANIFEST, MODEL_PERFORMANCE_DERIVED,
    UNKNOWN, InadmissibleFeature, UndeclaredFeature, check_admissible, classify,
    require_admissible,
)


def test_feature_provenance_manifest():
    """THE GUARDRAIL. Every feature the Governor sees is declared and admissible.

    A new feature added to richfeatures without a manifest entry fails here,
    which is the point: introducing benchmark metadata must be a deliberate,
    reviewed act rather than an import.
    """
    undeclared = [f for f in FEATURE_NAMES if f not in MANIFEST]
    assert not undeclared, (
        f"{len(undeclared)} decision-time feature(s) have no provenance entry: "
        f"{undeclared}. Declare them in governor/harness/provenance.py.")

    bad = check_admissible(FEATURE_NAMES)
    assert not bad, ("inadmissible features in the decision vector: "
                     + ", ".join(f"{p.feature}({p.cls})" for p in bad))


def test_unknown_is_rejected_not_defaulted():
    p = classify("some_feature_nobody_declared")
    assert p.cls == UNKNOWN
    assert not p.admissible()
    with pytest.raises(UndeclaredFeature):
        require_admissible(["some_feature_nobody_declared"])


def test_the_label_itself_is_refused():
    for f in ("hidden_all_passed", "hidden_pass_fraction", "grading_status",
              "private_test_cases"):
        assert classify(f).cls == HIDDEN_INFORMATION
        assert not classify(f).admissible()
    with pytest.raises(InadmissibleFeature):
        require_admissible(["hidden_all_passed"])


def test_published_model_verdict_is_refused():
    """`graded` is LiveCodeBench's own verdict for a model on that problem."""
    assert classify("graded").cls == MODEL_PERFORMANCE_DERIVED
    with pytest.raises(InadmissibleFeature):
        require_admissible(["graded"])


def test_metadata_is_refused_because_it_measured_as_the_label():
    """Emptiness matched pass/fail exactly: zero off-diagonal over 4000 rows."""
    assert not classify("metadata").admissible()


def test_difficulty_is_admissible_and_carries_its_audit():
    """E0030 audit: platform labels and setter point values, fixed at contest
    publication, independent of any model or of the hidden tests."""
    p = classify("difficulty")
    assert p.admissible(), "the E0030 audit cleared difficulty"
    assert p.cls in ADMISSIBLE
    assert p.audit, "an admissible benchmark field must cite its audit"
    assert "e0030" in p.audit.lower()


def test_every_entry_uses_a_real_class():
    for name, p in MANIFEST.items():
        assert p.cls in CLASSES, f"{name} has bogus class {p.cls}"
        assert p.source, f"{name} declares no source"
        assert p.fixed_when, f"{name} does not say when its value was fixed"


def test_admissible_set_excludes_unknown():
    assert UNKNOWN not in ADMISSIBLE
