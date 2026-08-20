"""Run a candidate against PUBLIC tests and report only what an agent could see.

The boundary this file defends: everything here is derived from
`public_test_cases`, which ship as plain JSON alongside the problem and are
visible to anyone attempting it. Nothing here touches `private_test_cases`
(zlib-encoded, the hidden set), the published `metadata` (whose emptiness is
exactly the label -- 1530 empty/PASS, 2470 non-empty/FAIL, zero off-diagonal),
or `graded_list`. Those belong to the evaluator and stay on the other side.

LiveCodeBench mixes two calling conventions and they cannot share a harness:

  stdin      atcoder, codeforces -- feed text, compare stdout
  functional leetcode -- construct Solution, call one method, compare its
             return value

The functional path is the awkward one. The method name is not recorded as a
field; it has to be read out of `starter_code`, and arguments arrive as one
JSON value per line rather than a single JSON array.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict

from governor.execfeedback.sandbox import Status, run

_DEF = re.compile(r"def\s+(\w+)\s*\(")

# The permitted observable feature set, frozen by the preregistration and
# checked by `oracle_leakage`. Adding a name here is a protocol change.
FEATURE_NAMES = (
    "compile_ok", "runtime_error", "timeout", "output_nonempty",
    "pub_passed", "pub_failed", "pub_frac", "pub_all_passed",
    "exec_latency_s", "output_truncated",
)


@dataclass(frozen=True)
class Feedback:
    """What running one candidate against the public tests revealed."""
    compile_ok: bool
    runtime_error: bool
    timeout: bool
    output_nonempty: bool
    pub_passed: int
    pub_failed: int
    pub_frac: float
    pub_all_passed: bool
    exec_latency_s: float
    output_truncated: bool
    n_tests: int
    statuses: tuple[str, ...]

    def features(self) -> dict[str, float]:
        d = asdict(self)
        return {k: float(d[k]) for k in FEATURE_NAMES}


def _normalise(s: str) -> str:
    """Compare the way a judge does: trailing whitespace is not a wrong answer."""
    return "\n".join(line.rstrip() for line in s.strip().splitlines())


def _match_functional(stdout: str, expected: str) -> bool:
    """Compare a returned value, tolerating JSON formatting differences."""
    got = _normalise(stdout)
    exp = _normalise(expected)
    if got == exp:
        return True
    try:
        return json.loads(got) == json.loads(exp)
    except (json.JSONDecodeError, ValueError):
        return False


def _functional_wrapper(code: str, starter: str, raw_input: str) -> str | None:
    """Build a driver that calls the one method the starter code declares."""
    m = _DEF.search(starter or "")
    if not m:
        return None
    method = m.group(1)

    args = []
    for line in raw_input.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            args.append(json.loads(line))
        except json.JSONDecodeError:
            args.append(line)

    return (
        "import json, sys\n"
        "from typing import *\n"
        "try:\n"
        "    from math import *\n"
        "    from collections import *\n"
        "    from itertools import *\n"
        "    from functools import *\n"
        "    from heapq import *\n"
        "    import bisect\n"
        "except Exception:\n"
        "    pass\n"
        f"{code}\n"
        f"_args = json.loads({json.dumps(json.dumps(args))})\n"
        f"_r = Solution().{method}(*_args)\n"
        "print(json.dumps(_r) if not isinstance(_r, str) else _r)\n"
    )


def evaluate(code: str, tests: list[dict], *, platform: str = "",
             starter_code: str = "", timeout_s: float = 6.0,
             mem_mb: int = 1024) -> Feedback:
    """Execute `code` against every public test and summarise the outcome."""
    from governor.execfeedback.sandbox import check_syntax

    syntax_ok, _ = check_syntax(code)
    if not syntax_ok:
        return Feedback(False, False, False, False, 0, len(tests), 0.0, False,
                        0.0, False, len(tests), ("COMPILE_ERROR",) * len(tests))

    passed = failed = 0
    latency = 0.0
    any_runtime_error = any_timeout = any_output = truncated = False
    statuses: list[str] = []

    for tc in tests:
        functional = tc.get("testtype") == "functional" or platform == "leetcode"
        if functional:
            wrapped = _functional_wrapper(code, starter_code, tc["input"])
            if wrapped is None:                 # no method to call: unrunnable
                statuses.append("COMPILE_ERROR")
                failed += 1
                continue
            r = run(wrapped, "", timeout_s=timeout_s, mem_mb=mem_mb)
        else:
            r = run(code, tc["input"], timeout_s=timeout_s, mem_mb=mem_mb)

        latency += r.latency_s
        truncated |= r.truncated
        any_output |= bool(r.stdout.strip())
        statuses.append(r.status)

        if r.status == Status.TIMEOUT:
            any_timeout = True
            failed += 1
            continue
        if r.status in (Status.RUNTIME_ERROR, Status.MEMORY, Status.LAUNCH_FAILED):
            any_runtime_error = True
            failed += 1
            continue

        hit = (_match_functional(r.stdout, tc["output"]) if functional
               else _normalise(r.stdout) == _normalise(tc["output"]))
        passed += 1 if hit else 0
        failed += 0 if hit else 1

    n = max(len(tests), 1)
    return Feedback(
        compile_ok=True,
        runtime_error=any_runtime_error,
        timeout=any_timeout,
        output_nonempty=any_output,
        pub_passed=passed,
        pub_failed=failed,
        pub_frac=passed / n,
        pub_all_passed=(passed == len(tests) and len(tests) > 0),
        exec_latency_s=latency,
        output_truncated=truncated,
        n_tests=len(tests),
        statuses=tuple(statuses),
    )
