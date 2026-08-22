"""Where a decision-time feature came from, declared before it may be used.

The trap catalogue already rejects features by NAME. That is necessary and not
sufficient: it caught `graded` only after someone thought to add the word, and
`difficulty` sits on the forbidden list for a reason that turned out not to
apply to LiveCodeBench at all. A name-based rule cannot distinguish a benchmark
field that is published with the problem from one computed from the outcomes
being predicted -- and those are the same word.

So every feature the Governor may see at decision time is classified here, with
its source, and `check_admissible` refuses anything unclassified. UNKNOWN is a
rejection, not a default.

THE DISTINCTION THAT MATTERS is not "is this metadata" but WHEN the value was
fixed relative to the outcome:

    fixed BEFORE any model touched the problem     -> usable
    computed FROM outcomes we are predicting       -> leakage

A human-assigned contest rating is prior information. A field derived from
model pass rates is the label wearing a different name, and the published
LiveCodeBench `metadata` emptiness already turned out to BE the label once here
(1530 empty/PASS, 2470 non-empty/FAIL, zero off-diagonal).
"""
from __future__ import annotations

from dataclasses import dataclass

# ---- classes, ordered from safest to most disqualifying ------------------
LEGITIMATE_PRIOR = "LEGITIMATE_PRIOR"
CONTEST_RATING_DERIVED = "CONTEST_RATING_DERIVED"
EXECUTION_OBSERVABLE = "EXECUTION_OBSERVABLE"
MODEL_PERFORMANCE_DERIVED = "MODEL_PERFORMANCE_DERIVED"
POSTHOC_EVALUATION_DERIVED = "POSTHOC_EVALUATION_DERIVED"
HIDDEN_INFORMATION = "HIDDEN_INFORMATION"
UNKNOWN = "UNKNOWN"

CLASSES = frozenset({
    LEGITIMATE_PRIOR, CONTEST_RATING_DERIVED, EXECUTION_OBSERVABLE,
    MODEL_PERFORMANCE_DERIVED, POSTHOC_EVALUATION_DERIVED, HIDDEN_INFORMATION,
    UNKNOWN,
})

# Admissible at decision time. UNKNOWN is deliberately absent: an unaudited
# feature is refused rather than assumed innocent.
ADMISSIBLE = frozenset({
    LEGITIMATE_PRIOR, CONTEST_RATING_DERIVED, EXECUTION_OBSERVABLE,
})


@dataclass(frozen=True)
class Provenance:
    feature: str
    cls: str
    source: str
    fixed_when: str
    audit: str = ""

    def admissible(self) -> bool:
        return self.cls in ADMISSIBLE


def _obs(name: str, what: str) -> Provenance:
    """Produced by running the candidate against PUBLIC tests, or by parsing the
    code the model itself emitted. Costs a sample to observe and reveals nothing
    the model has not already been given."""
    return Provenance(name, EXECUTION_OBSERVABLE, what,
                      "after the model generates, before the spend decision")


# ---- the manifest --------------------------------------------------------
# Static structure of the generated program. Derived from the model's own
# output by parsing it; no benchmark metadata involved.
_STATIC = {n: _obs(n, "AST/text of the generated code (governor.execfeedback."
                      "richfeatures.static_features)")
           for n in ("code_chars", "code_lines", "ast_nodes", "ast_depth",
                     "n_loops", "n_branches", "n_functions", "n_try",
                     "ast_call_nodes", "has_recursion", "parse_ok")}

# Trajectory of attempts so far, from PUBLIC-test execution only.
_TRAJ = {n: _obs(n, "public-test execution feedback (governor.execfeedback."
                    "publictests) aggregated over attempts so far")
         for n in ("attempt_idx", "attempts_left", "best_pub_frac",
                   "mean_pub_frac", "last_pub_frac", "pub_frac_trend",
                   "all_same_pub_frac", "any_compile_error",
                   "frac_runtime_error", "frac_timeout", "mean_latency",
                   "last_latency", "code_len_var", "n_distinct_outcomes")}

MANIFEST: dict[str, Provenance] = {**_STATIC, **_TRAJ}

# Benchmark metadata, audited individually. These are NOT in the feature vector
# today; they are classified so a future experiment cannot introduce one
# silently, and so a rejection carries its reason.
MANIFEST["difficulty"] = Provenance(
    "difficulty", CONTEST_RATING_DERIVED,
    "LiveCodeBench `difficulty`, from platform labels and setter point values: "
    "LeetCode's own Easy/Medium/Hard; AtCoder problem point values bucketed "
    "[0-200)/[200-400)/[400-500]; CodeForces rating brackets {800}/(800-1000]/"
    "(1000-1300]. arXiv:2403.07974 sec. 2.",
    "at or before contest publication",
    "results/e0030_difficulty_provenance.json (E0030 audit). Admissible, with "
    "the recorded caveat that CodeForces ratings are computed from HUMAN "
    "contestant performance post-contest -- human, not model, and independent "
    "of LiveCodeBench's hidden tests; 9 of 475 problems.")

MANIFEST["platform"] = Provenance(
    "platform", LEGITIMATE_PRIOR, "LiveCodeBench `platform`",
    "at problem publication")
MANIFEST["contest_date"] = Provenance(
    "contest_date", LEGITIMATE_PRIOR, "LiveCodeBench `contest_date`",
    "at problem publication")

for _n, _why in {
    "graded": "LiveCodeBench's published per-problem verdict for a model",
    "pass_rate": "aggregate model pass rate",
    "solve_rate": "aggregate model solve rate",
    "model_accuracy": "aggregate model accuracy",
}.items():
    MANIFEST[_n] = Provenance(_n, MODEL_PERFORMANCE_DERIVED, _why,
                              "after models were evaluated")

for _n, _why in {
    "hidden_all_passed": "private-test outcome -- this IS the label",
    "hidden_pass_fraction": "private-test outcome",
    "hidden_tests_passed": "private-test outcome",
    "hidden_tests_failed": "private-test outcome",
    "hidden_tests_total": "private-test suite size",
    "grading_status": "private-test grading verdict",
    "private_test_cases": "the hidden tests themselves",
    "expected_output": "reference outputs",
    "reference_solution": "the reference solution",
}.items():
    MANIFEST[_n] = Provenance(_n, HIDDEN_INFORMATION, _why,
                              "only after the outcome exists")

# LiveCodeBench's published `metadata` emptiness matched the pass/fail label
# exactly on inspection: 1530 empty/PASS, 2470 non-empty/FAIL, zero
# off-diagonal. It looks like innocuous bookkeeping and is the label.
MANIFEST["metadata"] = Provenance(
    "metadata", POSTHOC_EVALUATION_DERIVED,
    "LiveCodeBench `metadata`; its emptiness matched the label exactly "
    "(1530 empty/PASS, 2470 non-empty/FAIL, zero off-diagonal)",
    "after evaluation", "measured directly on the E0026 pull")


class UndeclaredFeature(RuntimeError):
    """A decision-time feature with no provenance entry. Refused, not assumed."""


class InadmissibleFeature(RuntimeError):
    """A declared feature whose class disqualifies it from decision time."""


def classify(feature: str) -> Provenance:
    return MANIFEST.get(feature, Provenance(feature, UNKNOWN, "not declared",
                                            "unknown"))


def check_admissible(features) -> list[Provenance]:
    """Return the offending entries; empty means every feature is cleared.

    Callers that must not proceed should use `require_admissible`.
    """
    return [p for p in (classify(f) for f in features) if not p.admissible()]


def require_admissible(features) -> None:
    bad = check_admissible(features)
    if not bad:
        return
    undeclared = [p for p in bad if p.cls == UNKNOWN]
    if undeclared:
        raise UndeclaredFeature(
            "features with no provenance entry: "
            + ", ".join(p.feature for p in undeclared)
            + "\n  Declare them in governor/harness/provenance.py with a source "
              "and the time their value was fixed. UNKNOWN is a rejection, not "
              "a default -- an unaudited feature is how benchmark metadata "
              "enters an experiment silently.")
    raise InadmissibleFeature(
        "features disqualified at decision time:\n"
        + "\n".join(f"    {p.feature}: {p.cls}\n      {p.source}" for p in bad))
