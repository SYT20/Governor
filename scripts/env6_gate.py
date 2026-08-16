#!/usr/bin/env python3
"""Gates 1-3 on Environment 6. Every policy runs through the canonical executor.

GATE 1  best constant schedule chosen on CALIBRATION only, then frozen
GATE 2  adaptive oracle -- full state information, exhaustive over which 2 of 4
        items receive the deep call (6 patterns, small enough to enumerate)
GATE 3  M1 = U_oracle - U_constant  must exceed EPSILON = 0.02 (FROZEN)
        M2 = P(per-episode difference > EPSILON)
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from governor.gate.env6 import Env6  # noqa: E402
from governor.gate.executor import run_episode  # noqa: E402

EPSILON, BUDGET = 0.02, 2.0
N_CAL, N_TEST = 500, 500


def const_policy(slots):
    def p(o, left):
        return "M2" if (o["t"] in slots and left >= 1.0) else "H"
    return p


def cue_policy(o, left):
    """OBSERVABLE adaptive reference: spend on items the cue calls hard."""
    return "M2" if (o["cue"] == 1 and left >= 1.0) else "H"


def oracle_policy(env, ep):
    """GATE 2: the DIFFICULTY oracle. Knows hidden difficulty, NOT the outcome.

    A first version took max over the 6 patterns of REALISED utility, which
    picks whichever pattern happened to win given the pre-drawn rolls. That is
    clairvoyant about outcomes, not difficulty -- recorded failure mode #4,
    realised stochastic outcomes treated as expected value, and it inflates the
    ceiling to something no policy could ever reach.

    The correct oracle maximises EXPECTED accuracy given difficulty: place the
    two deep calls on the items with the largest expected gain, which is the
    hard ones (0.85-0.50 = 0.35) over the easy ones (0.95-0.90 = 0.05).
    Ties broken by index, deterministically.
    """
    hard = env.hard[ep]
    gain = [0.35 if h else 0.05 for h in hard]
    slots = set(sorted(range(4), key=lambda i: (-gain[i], i))[:2])
    return const_policy(slots)


def mean_u(env, pol, eps):
    return float(np.mean([run_episode(env, pol, e, BUDGET).utility for e in eps]))


def main() -> int:
    print("=" * 72)
    print("ENVIRONMENT 6 — GATES 1-3   (EPSILON = 0.02, frozen)")
    print("=" * 72)
    env = Env6(seed=0, n=N_CAL + N_TEST)
    cal, test = range(N_CAL), range(N_CAL, N_CAL + N_TEST)

    scheds = [s for k in range(3) for s in itertools.combinations(range(4), k)]
    scored = sorted(((mean_u(env, const_policy(set(s)), cal), s)
                     for s in scheds), reverse=True)
    best_s = scored[0][1]
    print(f"\nGATE 1  {len(scheds)} constant schedules on CALIBRATION")
    for u, s in scored[:4]:
        print(f"    {str(sorted(s)):<12} U={u:.4f}")
    print(f"  frozen best constant: {sorted(best_s)}")

    u_const = mean_u(env, const_policy(set(best_s)), test)
    u_cue = mean_u(env, cue_policy, test)
    u_or = float(np.mean([run_episode(env, oracle_policy(env, e), e, BUDGET).utility
                          for e in test]))
    per = np.array([run_episode(env, oracle_policy(env, e), e, BUDGET).utility
                    - run_episode(env, const_policy(set(best_s)), e, BUDGET).utility
                    for e in test])
    se = per.std(ddof=1) / np.sqrt(len(per))
    m1, m2 = u_or - u_const, float(np.mean(per > EPSILON))

    print(f"\nGATE 2/3  HELD-OUT ({N_TEST} episodes)")
    print(f"    U_constant (frozen)     {u_const:.4f}")
    print(f"    U_cue (observable)      {u_cue:.4f}")
    print(f"    U_oracle (difficulty)   {u_or:.4f}")
    print(f"    M1 = oracle - constant  {m1:+.4f}  [{m1-1.96*se:+.4f}, {m1+1.96*se:+.4f}]")
    print(f"    M2 = P(diff > {EPSILON})     {m2:.0%}")
    print(f"    observable headroom     {u_cue - u_const:+.4f}  "
          f"({(u_cue-u_const)/max(m1,1e-9):.0%} of oracle)")

    ok = m1 > EPSILON and m1 - 1.96 * se > 0
    print(f"\n  GATE 3: {'PASS -- environment contains adaptive headroom' if ok else 'FAIL -- reject environment'}")
    Path("results").mkdir(exist_ok=True)
    Path("results/env6_gate.json").write_text(json.dumps(
        {"u_const": u_const, "u_cue": u_cue, "u_oracle": u_or, "m1": m1,
         "m2": m2, "best_constant": list(best_s), "pass": bool(ok)}, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
