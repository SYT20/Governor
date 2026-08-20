"""Preflight for any generation backend. The token budget is part of the instrument.

E0029 was stopped after 76 of 4750 samples because a `max_completion_tokens` cap
of 2500 was manufacturing failures:

    at the cap     26 samples,  11 empty (42%)
    under the cap  50 samples,   0 empty (0%)

Perfect separation. A reasoning model spends most of its budget reasoning before
emitting anything, so a third of samples were cut off mid-thought. Those would
have been scored as the model failing the problem when the configuration failed
instead -- and because harder problems reason longer, the artificial failure rate
would have risen with difficulty, which is precisely the signal the allocation
experiment tries to detect.

Thirty hours of generation would have produced a confounded replication. A
two-minute check would have caught it. So this is now a gate, not a courtesy:

    preflight -> distribution, truncation, empty output, variance -> full run

FOUR CHECKS, each with a reason to exist:

  truncation  a capped completion is not a sample of the model, it is a sample
              of the cap
  empty       output that yields nothing to run is scored as failure regardless
              of ability
  headroom    a backend that solves everything or nothing has no allocation
              problem to study, whatever its accuracy
  latency     projects the full run honestly, since every previous estimate here
              was wrong -- 388, then 1829, then 2378 tokens per sample
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field, asdict

MAX_TRUNCATION_RATE = 0.05
MAX_EMPTY_RATE = 0.02
MIN_SOLVE_RATE = 0.05
MAX_SOLVE_RATE = 0.95


@dataclass
class Sample:
    """One preflight generation."""
    completion_tokens: int
    truncated: bool
    code: str
    latency_s: float
    solved: bool | None = None          # public-test outcome, if measured


@dataclass
class PreflightReport:
    n: int
    truncation_rate: float
    empty_rate: float
    median_completion: float
    p95_completion: float
    max_completion: float
    median_latency: float
    solve_rate: float | None
    ok: bool
    problems: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)

    def render(self) -> str:
        mark = "PASS" if self.ok else "FAIL"
        lines = [
            f"  PREFLIGHT {mark}   n={self.n}",
            f"    completion tokens   median {self.median_completion:.0f}  "
            f"p95 {self.p95_completion:.0f}  max {self.max_completion:.0f}",
            f"    truncation rate     {self.truncation_rate:.1%}   (limit {MAX_TRUNCATION_RATE:.0%})",
            f"    empty-output rate   {self.empty_rate:.1%}   (limit {MAX_EMPTY_RATE:.0%})",
            f"    median latency      {self.median_latency:.2f}s",
        ]
        if self.solve_rate is not None:
            lines.append(f"    public solve rate   {self.solve_rate:.1%}   "
                         f"(need {MIN_SOLVE_RATE:.0%}-{MAX_SOLVE_RATE:.0%} for headroom)")
        for p in self.problems:
            lines.append(f"    PROBLEM: {p}")
        return "\n".join(lines)

    def project(self, total_samples: int, tokens_per_min: float | None = None) -> str:
        """Honest projection of the full run, from measured rates."""
        tok = self.median_completion * total_samples
        wall = self.median_latency * total_samples / 3600
        out = [f"    projected {total_samples} samples: {tok/1e6:.2f}M completion tokens",
               f"      latency-bound: {wall:.1f} h"]
        if tokens_per_min:
            out.append(f"      rate-limit-bound at {tokens_per_min:.0f}/min: "
                       f"{tok/tokens_per_min/60:.1f} h")
        return "\n".join(out)


def assess(samples: list[Sample], *, cap: int) -> PreflightReport:
    """Decide whether a full run on this backend and cap would measure the model."""
    n = len(samples)
    if n == 0:
        return PreflightReport(0, 1.0, 1.0, 0, 0, 0, 0, None, False,
                               ["no preflight samples"])

    comp = [s.completion_tokens for s in samples]
    trunc = sum(1 for s in samples if s.truncated or s.completion_tokens >= cap - 10) / n
    empty = sum(1 for s in samples if not s.code.strip()) / n
    graded = [s.solved for s in samples if s.solved is not None]
    solve = (sum(graded) / len(graded)) if graded else None

    problems: list[str] = []
    if trunc > MAX_TRUNCATION_RATE:
        problems.append(
            f"truncation {trunc:.0%} exceeds {MAX_TRUNCATION_RATE:.0%} -- raise the cap; "
            f"a capped completion samples the cap, not the model")
    if empty > MAX_EMPTY_RATE:
        problems.append(
            f"empty output {empty:.0%} exceeds {MAX_EMPTY_RATE:.0%} -- these score as "
            f"failures regardless of ability")
    if solve is not None and solve < MIN_SOLVE_RATE:
        problems.append(f"solve rate {solve:.0%} below {MIN_SOLVE_RATE:.0%} -- "
                        f"nothing to allocate between")
    if solve is not None and solve > MAX_SOLVE_RATE:
        problems.append(f"solve rate {solve:.0%} above {MAX_SOLVE_RATE:.0%} -- "
                        f"no headroom for allocation to exploit")

    return PreflightReport(
        n=n, truncation_rate=trunc, empty_rate=empty,
        median_completion=float(statistics.median(comp)),
        p95_completion=float(sorted(comp)[min(n - 1, int(0.95 * n))]),
        max_completion=float(max(comp)),
        median_latency=float(statistics.median(s.latency_s for s in samples)),
        solve_rate=solve, ok=not problems, problems=problems)


class PreflightFailed(RuntimeError):
    """Raised instead of letting a confounded run proceed."""


def require(report: PreflightReport) -> None:
    if not report.ok:
        raise PreflightFailed(
            "preflight failed; a full run would measure the configuration rather "
            "than the model:\n" + "\n".join(f"  - {p}" for p in report.problems))
