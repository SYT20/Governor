"""Decode and run LiveCodeBench's PRIVATE tests. Evaluator-only, by construction.

This module produces the hidden outcome label. It is deliberately separate from
`publictests.py`, which produces the observable features, because the entire
experiment rests on those two never mixing:

    PUBLIC execution  ->  observable features  ->  Governor decision
    PRIVATE execution ->  outcome label        ->  scoring, after the decision

Nothing here may ever be read at decision time. `FORBIDDEN_AS_FEATURE` names the
fields this module emits so the analysis can fail closed on any attempt, rather
than relying on a convention someone has to remember.

WHY THIS EXISTS AT ALL: E0029 generated 4750 Qwen samples carrying only
public-test outcomes. Without an independent hidden label, utility could only be
defined from the same signal the features come from -- perfect leakage, and the
precise failure the preregistration forbade after LiveCodeBench's published
metadata turned out to *be* the label.

DECODING UNTRUSTED BYTES: the private cases are base64 -> zlib -> pickle, and
pickle executes arbitrary code on load. The source is a specific, pinned
benchmark artifact rather than arbitrary remote input, but the decoded object is
still validated structurally before use, and a decode that yields anything other
than the expected shape is rejected rather than trusted.
"""
from __future__ import annotations

import base64
import json
import pickle
import time
import zlib
from dataclasses import dataclass, asdict

from governor.execfeedback.publictests import _match_functional, _normalise
from governor.execfeedback.sandbox import Status, check_syntax, run

# Emitted by this module. The analysis rejects any feature drawn from these.
FORBIDDEN_AS_FEATURE = (
    "hidden_tests_total", "hidden_tests_passed", "hidden_tests_failed",
    "hidden_pass_fraction", "hidden_all_passed", "grading_status",
    "private_test_cases", "private_expected",
)


class PrivateTestError(RuntimeError):
    """Raised rather than returning something that merely looks like tests."""


@dataclass(frozen=True)
class Grade:
    """The hidden outcome for one generated sample."""
    hidden_tests_total: int
    hidden_tests_passed: int
    hidden_tests_failed: int
    hidden_pass_fraction: float
    hidden_all_passed: bool
    grading_status: str
    grading_latency_ms: int

    def as_dict(self) -> dict:
        return asdict(self)


def decode_private_tests(raw: str) -> list[dict]:
    """base64 -> zlib -> pickle -> (JSON if a string), then validate the shape.

    Returns the test cases. Raises `PrivateTestError` with the failing stage
    rather than propagating a decoder's own exception, which would name a codec
    rather than the data.
    """
    if not isinstance(raw, str) or not raw:
        raise PrivateTestError("private_test_cases is empty or not a string")
    try:
        blob = base64.b64decode(raw)
    except Exception as e:                                # noqa: BLE001
        raise PrivateTestError(f"base64 stage: {type(e).__name__}") from e
    try:
        blob = zlib.decompress(blob)
    except Exception as e:                                # noqa: BLE001
        raise PrivateTestError(f"zlib stage: {type(e).__name__}") from e
    try:
        obj = pickle.loads(blob)                          # noqa: S301
    except Exception as e:                                # noqa: BLE001
        raise PrivateTestError(f"pickle stage: {type(e).__name__}") from e
    if isinstance(obj, (str, bytes)):
        try:
            obj = json.loads(obj)
        except Exception as e:                            # noqa: BLE001
            raise PrivateTestError(f"json stage: {type(e).__name__}") from e

    if not isinstance(obj, list) or not obj:
        raise PrivateTestError(
            f"expected a non-empty list of cases, got {type(obj).__name__}")
    for i, tc in enumerate(obj):
        if not isinstance(tc, dict):
            raise PrivateTestError(f"case {i} is {type(tc).__name__}, not a dict")
        missing = [k for k in ("input", "output") if k not in tc]
        if missing:
            raise PrivateTestError(f"case {i} lacks {missing}")
    return obj


def grade(code: str, private_tests: list[dict], *, platform: str = "",
          starter_code: str = "", timeout_s: float = 6.0,
          mem_mb: int = 1024, max_tests: int | None = None) -> Grade:
    """Run one candidate against the hidden tests and report the outcome.

    Never raises for a misbehaving program. A crash, a hang and an allocation
    storm are outcomes to be recorded; one bad generated program must not end a
    grading job of thousands.
    """
    t0 = time.perf_counter()
    tests = private_tests[:max_tests] if max_tests else private_tests
    n = len(tests)

    def done(passed: int, status: str) -> Grade:
        return Grade(
            hidden_tests_total=n, hidden_tests_passed=passed,
            hidden_tests_failed=n - passed,
            hidden_pass_fraction=(passed / n) if n else 0.0,
            hidden_all_passed=(passed == n and n > 0), grading_status=status,
            grading_latency_ms=int((time.perf_counter() - t0) * 1000))

    if not code or not code.strip():
        return done(0, "EMPTY_CODE")
    ok, _ = check_syntax(code)
    if not ok:
        return done(0, "COMPILE_ERROR")

    from governor.execfeedback.publictests import _functional_wrapper

    passed, statuses = 0, set()
    for tc in tests:
        functional = tc.get("testtype") == "functional" or platform == "leetcode"
        try:
            if functional:
                wrapped = _functional_wrapper(code, starter_code, tc["input"])
                if wrapped is None:
                    statuses.add("NO_ENTRYPOINT")
                    continue
                r = run(wrapped, "", timeout_s=timeout_s, mem_mb=mem_mb)
            else:
                r = run(code, tc["input"], timeout_s=timeout_s, mem_mb=mem_mb)
        except Exception as e:                            # noqa: BLE001
            statuses.add(f"HARNESS_{type(e).__name__}")
            continue

        statuses.add(r.status)
        if r.status != Status.OK:
            continue
        hit = (_match_functional(r.stdout, tc["output"]) if functional
               else _normalise(r.stdout) == _normalise(tc["output"]))
        passed += 1 if hit else 0

    if passed == n and n > 0:
        return done(passed, "OK")
    if Status.TIMEOUT in statuses:
        return done(passed, "TIMEOUT")
    if Status.RUNTIME_ERROR in statuses or Status.MEMORY in statuses:
        return done(passed, "RUNTIME_ERROR")
    if any(s.startswith("HARNESS_") for s in statuses):
        return done(passed, "HARNESS_ERROR")
    return done(passed, "WRONG_ANSWER")
