"""Richer decision-time features for E0028, and the boundary they respect.

E0027 diagnosed the learned ranker as data-starved rather than weak: the target
"sample 1 failed and a later one succeeds" occurs 19 times in 207 calibration
problems, 2.7 events per feature. Two changes follow, and both are about getting
more information out of the same 400 problems rather than getting more problems:

  TARGET   ask, at each decision point, whether ANY later sample succeeds given
           every attempt so far has failed. 1003 rows, 49 positives -- 2.6x, and
           7.0 events per feature.

  FEATURES describe the TRAJECTORY of attempts, not just the last one. Whether
           three attempts all failed the same public test is different evidence
           from three attempts failing different ones, and the old feature set
           could not express the difference.

EVERYTHING HERE IS AVAILABLE AT DECISION TIME. The state after drawing samples
0..i consists of those samples' source code and what happened when each was run
against public tests. Nothing reads hidden grades, private tests, LiveCodeBench's
metadata, or any sample not yet drawn.

Static code properties are free: they need no re-execution, only the source that
was already generated.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass

STATIC_NAMES = (
    "code_chars", "code_lines", "ast_nodes", "ast_depth",
    "n_loops", "n_branches", "n_functions", "n_try", "n_calls",
    "has_recursion", "parse_ok",
)

TRAJECTORY_NAMES = (
    "attempt_idx", "attempts_left",
    "best_pub_frac", "mean_pub_frac", "last_pub_frac", "pub_frac_trend",
    "all_same_pub_frac", "any_compile_error", "frac_runtime_error",
    "frac_timeout", "mean_latency", "last_latency",
    "code_len_var", "n_distinct_outcomes",
)

FEATURE_NAMES = STATIC_NAMES + TRAJECTORY_NAMES


@dataclass(frozen=True)
class _Static:
    values: dict[str, float]


def static_features(code: str) -> dict[str, float]:
    """Structure of the generated program. No execution required."""
    out = {n: 0.0 for n in STATIC_NAMES}
    out["code_chars"] = float(len(code))
    out["code_lines"] = float(code.count("\n") + 1)
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError, MemoryError, RecursionError):
        out["parse_ok"] = 0.0
        return out
    out["parse_ok"] = 1.0

    nodes = 0
    fn_names: set[str] = set()
    called: set[str] = set()
    max_depth = 0

    def walk(node, depth):
        nonlocal nodes, max_depth
        nodes += 1
        max_depth = max(max_depth, depth)
        if isinstance(node, (ast.For, ast.While, ast.comprehension)):
            out["n_loops"] += 1
        elif isinstance(node, (ast.If, ast.IfExp)):
            out["n_branches"] += 1
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out["n_functions"] += 1
            fn_names.add(node.name)
        elif isinstance(node, (ast.Try, ast.ExceptHandler)):
            out["n_try"] += 1
        elif isinstance(node, ast.Call):
            out["n_calls"] += 1
            if isinstance(node.func, ast.Name):
                called.add(node.func.id)
        for child in ast.iter_child_nodes(node):
            walk(child, depth + 1)

    try:
        walk(tree, 0)
    except RecursionError:
        pass
    out["ast_nodes"] = float(nodes)
    out["ast_depth"] = float(max_depth)
    out["has_recursion"] = 1.0 if (fn_names & called) else 0.0
    return out


def trajectory_features(attempts: list[dict]) -> dict[str, float]:
    """What the sequence of attempts so far looks like.

    `attempts` are the execution-feedback records for samples 0..i, all of which
    failed. The caller must not include the sample being decided about.
    """
    out = {n: 0.0 for n in TRAJECTORY_NAMES}
    if not attempts:
        return out
    n = len(attempts)
    fr = [float(a["pub_frac"]) for a in attempts]
    lat = [float(a["exec_latency_s"]) for a in attempts]
    lens = [float(a.get("code_chars", 0.0)) for a in attempts]

    out["attempt_idx"] = float(n)
    out["attempts_left"] = float(max(0, 10 - n))
    out["best_pub_frac"] = max(fr)
    out["mean_pub_frac"] = sum(fr) / n
    out["last_pub_frac"] = fr[-1]
    out["pub_frac_trend"] = (fr[-1] - fr[0]) if n > 1 else 0.0
    # Repeated identical failure is evidence the model is stuck, which is
    # different from varied failure even at the same average score.
    out["all_same_pub_frac"] = 1.0 if len(set(fr)) == 1 else 0.0
    out["any_compile_error"] = 1.0 if any(a["compile_ok"] < 0.5 for a in attempts) else 0.0
    out["frac_runtime_error"] = sum(a["runtime_error"] > 0.5 for a in attempts) / n
    out["frac_timeout"] = sum(a["timeout"] > 0.5 for a in attempts) / n
    out["mean_latency"] = sum(lat) / n
    out["last_latency"] = lat[-1]
    out["code_len_var"] = float(max(lens) - min(lens)) if lens else 0.0
    out["n_distinct_outcomes"] = float(len({(a["pub_passed"], a["pub_failed"],
                                             a["runtime_error"], a["timeout"])
                                            for a in attempts}))
    return out


def decision_features(attempts: list[dict]) -> dict[str, float]:
    """The full decision-time vector: last attempt's structure plus the trajectory."""
    last = attempts[-1] if attempts else {}
    static = {k: float(last.get(k, 0.0)) for k in STATIC_NAMES}
    return {**static, **trajectory_features(attempts)}
