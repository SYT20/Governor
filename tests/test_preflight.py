"""The preflight gate must reject the exact configuration that stopped E0029.

E0029 was halted after 76 of 4750 samples: a 2500-token cap truncated 34% of
completions and 42% of those produced no code at all, while 0% of uncapped ones
did. A thirty-hour run would have measured the cap instead of the model.
"""
from __future__ import annotations

import pytest

from governor.execfeedback.preflight import (
    MAX_EMPTY_RATE, MAX_TRUNCATION_RATE, PreflightFailed, PreflightReport,
    Sample, assess, require,
)


def _s(tokens, code="print(1)", cap=2500, latency=1.0, solved=None):
    return Sample(completion_tokens=tokens, truncated=tokens >= cap,
                  code=code, latency_s=latency, solved=solved)


def test_rejects_the_e0029_configuration():
    """The real numbers: 26 of 76 at the cap, 11 of those empty."""
    samples = ([_s(2500, code="") for _ in range(11)]
               + [_s(2500) for _ in range(15)]
               + [_s(900) for _ in range(50)])
    r = assess(samples, cap=2500)
    assert not r.ok
    assert r.truncation_rate > MAX_TRUNCATION_RATE
    assert r.empty_rate > MAX_EMPTY_RATE
    with pytest.raises(PreflightFailed, match="truncation"):
        require(r)


def test_accepts_a_healthy_backend():
    samples = [_s(700 + i, solved=(i % 3 == 0)) for i in range(40)]
    r = assess(samples, cap=2500)
    assert r.ok, r.problems
    require(r)


def test_flags_a_backend_that_solves_nothing():
    """Accuracy is not the point; an allocation problem needs variance."""
    r = assess([_s(800, solved=False) for _ in range(30)], cap=2500)
    assert not r.ok
    assert any("solve rate" in p for p in r.problems)


def test_flags_a_backend_that_solves_everything():
    r = assess([_s(800, solved=True) for _ in range(30)], cap=2500)
    assert not r.ok
    assert any("no headroom" in p for p in r.problems)


def test_empty_preflight_is_a_failure_not_a_pass():
    """Missing evidence must never read as success -- the harness rule."""
    r = assess([], cap=2500)
    assert not r.ok


def test_truncation_counted_just_below_the_cap():
    """A completion stopping at cap-1 is truncated too."""
    r = assess([_s(2495) for _ in range(20)], cap=2500)
    assert r.truncation_rate == 1.0


def test_projection_uses_measured_rates():
    r = assess([_s(1000, latency=2.0) for _ in range(20)], cap=2500)
    text = r.project(4750, tokens_per_min=8000)
    assert "M completion tokens" in text and "h" in text


def test_report_is_serialisable_for_provenance():
    r = assess([_s(900, solved=True) for _ in range(10)]
               + [_s(900, solved=False) for _ in range(10)], cap=2500)
    d = r.as_dict()
    assert d["n"] == 20 and "truncation_rate" in d and isinstance(d["ok"], bool)
