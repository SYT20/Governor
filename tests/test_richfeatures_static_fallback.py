"""Static features must survive a row that carries `code` but no AST columns.

E0029's generation rows are exactly that shape. decision_features used to read
every static name with `.get(k, 0.0)`, so all eleven silently became zero and an
experiment built around static code structure ran with none of it. Nothing
raised. The only symptom was a diagnostic reporting 17 of 25 features constant.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from governor.execfeedback.richfeatures import STATIC_NAMES, decision_features

BASE = {"pub_frac": 0.5, "exec_latency_s": 0.1, "compile_ok": 1.0,
        "runtime_error": 0.0, "timeout": 0.0, "pub_passed": 1, "pub_failed": 1}

SHORT = "print(1)\n"
LONG = ("import sys\n"
        "def solve(n):\n"
        "    total = 0\n"
        "    for i in range(n):\n"
        "        if i % 2:\n"
        "            total += i\n"
        "    return total\n"
        "print(solve(10))\n")


def _feat(code):
    return decision_features([{**BASE, "code": code}])


def test_static_features_derived_from_code_when_columns_absent():
    a, b = _feat(SHORT), _feat(LONG)
    assert a["code_chars"] == float(len(SHORT))
    assert b["code_chars"] == float(len(LONG))
    assert b["ast_nodes"] > a["ast_nodes"]
    assert b["n_loops"] == 1.0 and a["n_loops"] == 0.0
    assert b["n_branches"] == 1.0 and a["n_branches"] == 0.0
    assert b["n_functions"] == 1.0


def test_static_features_are_not_all_zero():
    f = _feat(LONG)
    assert any(f[k] != 0.0 for k in STATIC_NAMES)
    # the specific regression: every static name zeroed at once
    assert not all(f[k] == 0.0 for k in STATIC_NAMES)


def test_precomputed_columns_win_over_code():
    """E0028's rows carry the AST columns already; they must not be recomputed."""
    f = decision_features([{**BASE, "code": SHORT, "code_chars": 999.0,
                            "ast_nodes": 42.0}])
    assert f["code_chars"] == 999.0
    assert f["ast_nodes"] == 42.0


def test_code_len_var_uses_code_when_column_absent():
    f = decision_features([{**BASE, "code": SHORT}, {**BASE, "code": LONG}])
    assert f["code_len_var"] == float(len(LONG) - len(SHORT))


def test_no_code_and_no_columns_is_zero_not_a_crash():
    f = decision_features([dict(BASE)])
    assert all(f[k] == 0.0 for k in STATIC_NAMES)
