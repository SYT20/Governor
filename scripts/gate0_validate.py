#!/usr/bin/env python3
"""Run Gate 0 + Gate 0B: canonical executor, then the positive control.

Every policy below -- constant schedules, adaptive oracle -- is scored by
run_episode and nothing else. If the harness cannot detect the constructed
adaptive advantage, the harness is invalid and no later rejection means
anything.
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from governor.gate.executor import run_episode  # noqa: E402
from governor.gate.positive_control import PositiveControlEnv  # noqa: E402

EPSILON = 0.02          # FROZEN materiality threshold
BUDGET, N_CAL, N_TEST = 2.0, 200, 200


def const_policy(slots):
    def p(obs, left):
        return "M2" if (obs["t"] in slots and left >= 1.0) else "H"
    return p


def adaptive_policy(obs, left):
    """Reads the signal. Spends its one useful call at the revealed target."""
    if obs["signal"] == -1:
        return "H"                       # signal not yet revealed
    return "M2" if obs["t"] == obs["signal"] and left >= 1.0 else "H"


def mean_u(env, pol, eps):
    return float(np.mean([run_episode(env, pol, e, BUDGET).utility for e in eps]))


def main() -> int:
    print("=" * 70)
    print("GATE 0 (canonical executor) + GATE 0B (positive control)")
    print("=" * 70)
    env = PositiveControlEnv(seed=0, n=N_CAL + N_TEST)
    cal, test = range(N_CAL), range(N_CAL, N_CAL + N_TEST)

    # Gate 1: best constant schedule chosen on CALIBRATION only
    schedules = [s for k in range(3)
                 for s in itertools.combinations(range(4), k)]
    scored = [(mean_u(env, const_policy(set(s)), cal), s) for s in schedules]
    best_u_cal, best_s = max(scored)
    print(f"\n  {len(schedules)} constant schedules scored on calibration")
    for u, s in sorted(scored, reverse=True)[:4]:
        print(f"    {str(set(s) or '{}'):<12} U={u:+.4f}")
    print(f"  best constant (frozen): {set(best_s) or '{}'}")

    # Gates 1-3 evaluated on HELD-OUT
    u_const = mean_u(env, const_policy(set(best_s)), test)
    u_adapt = mean_u(env, adaptive_policy, test)
    m1 = u_adapt - u_const
    per = np.array([run_episode(env, adaptive_policy, e, BUDGET).utility
                    - run_episode(env, const_policy(set(best_s)), e, BUDGET).utility
                    for e in test])
    se = per.std(ddof=1) / np.sqrt(len(per))
    m2 = float(np.mean(per > EPSILON))

    print(f"\n  HELD-OUT ({N_TEST} episodes)")
    print(f"    U_constant  {u_const:+.4f}")
    print(f"    U_adaptive  {u_adapt:+.4f}")
    print(f"    M1 headroom {m1:+.4f}  [{m1-1.96*se:+.4f}, {m1+1.96*se:+.4f}]")
    print(f"    M2 prevalence P(diff > {EPSILON}) = {m2:.0%}")

    ok = m1 > EPSILON and m1 - 1.96 * se > 0
    print(f"\n  GATE 0B: {'PASS -- harness can detect adaptive advantage'
                        if ok else 'FAIL -- HARNESS INVALID, do not trust rejections'}")
    Path("results").mkdir(exist_ok=True)
    Path("results/gate0_validate.json").write_text(json.dumps(
        {"u_const": u_const, "u_adapt": u_adapt, "m1": m1, "m2": m2,
         "best_constant": list(best_s), "pass": bool(ok)}, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
