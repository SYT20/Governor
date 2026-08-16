# Construction gate for the next environment — run BEFORE any Governor exists

Derived from nine negative results. Environments 1–5 each passed some check and
then collapsed; this gate is the union of what would have caught them.

## The prerequisite Env 5 was missing

Env 5 has genuine non-myopic value, genuine state variation, and a working
state-conditioned H-vs-M2 selector. It still failed, because when compute was
made scarce **the optimal allocation collapsed to a constant temporal rule**
("never spend early"). Measured signed Δ\* by decision point:

    t=0  -0.1250  (SD exactly 0.0000 — the state IS the prior)
    t=1  -0.0119
    t=2  +0.0025   positive in  8% of decisions
    t=3  +0.0050   positive in 16%

`P(Top2 ≠ {0,1}) = 100%` looked like a large allocation effect. It was entirely
"don't spend at t=0", which needs no state information.

> **Reasoning has value** and **reasoning needs adaptive allocation** are
> different claims. Every environment so far demonstrated the first and failed
> the second.

## Five requirements, all measured before implementation

1. **State-dependent sign.** Within one regime and budget,
   `P(Δ* > 0) > 0` **and** `P(Δ* < 0) > 0`.
2. **No trivial temporal schedule.** The best allocation must not be
   "always first k", "always last k", or "never first".
3. **Genuine scarcity.** Always-deliberate must be infeasible or lose, because
   the budget is shared across decisions.
4. **Multiple optimal patterns.** Within one episode family, the preferred
   allocation must differ across episodes.
5. **Observable predictability.** The preferred allocation must be predictable
   from information the controller has — not from hidden configuration id.

## The two telemetry metrics that decide it

**M1 (PRIMARY) — adaptive allocation headroom**

    U(oracle-adaptive) - U(best-constant-schedule)   on held-out episodes

**M2 (SECONDARY) — prevalence of that headroom**

    P( U(oracle-adaptive) - U(best-constant-schedule) > epsilon )
    for a preregistered epsilon, over held-out episodes

An earlier draft made M1 "number of distinct optimal schedules". That is wrong:
ties and numerical noise produce many distinct optimal schedules while
`U_adaptive ~= U_constant`, so schedule multiplicity would register as adaptive
value when there is none. Both metrics are now stated as VALUE differences, and
the prevalence form replaces the diversity count.

**Reject the environment if the best constant schedule captures essentially all
available value.** That single comparison is what Env 5 lacked, and it is cheap
— no learned model, no executor, no Governor.

## Execution protocol, in order

    Gate 0  one canonical episode executor, shared by constant schedules, the
            adaptive oracle, and eventually the Governor. Nothing may score a
            policy except by running it through this. Run I failed precisely
            because scoring and execution were separate paths.
    Gate 0b POSITIVE CONTROL. Before trusting the gate, run it on a toy
            environment where adaptive allocation beats every constant schedule
            BY CONSTRUCTION -- e.g. a signal at t=2 that says whether the
            valuable call is at t=3 or never. The gate must report material M1
            there. Without this, a buggy gate and a degenerate environment
            produce identical output, and every rejection is unfalsifiable.
    Gate 1  enumerate every feasible constant schedule -> U(best-constant).
            No model, no training, no hidden configuration.

            THE CONSTANT BASELINE MUST BE CHOSEN ON A CALIBRATION SPLIT.
            Running all constant schedules on the TEST episodes and keeping the
            best makes the test set part of schedule selection, which inflates
            nothing about the adaptive arm but deflates the baseline -- so it
            manufactures headroom. Fit the schedule on calibration episodes,
            freeze it, then evaluate that frozen schedule on held-out. Same
            discovery/evaluation separation applied to configurations earlier.
    Gate 2  solve with full state information -> U(oracle-adaptive).
            Brute-force the policy tree while it is small; exact DP if not.
    Gate 3  reject unless M1 headroom is material. This is the gate that would
            have closed Env 5 in twenty minutes instead of nine iterations.
    Gate 4  re-measure on fresh seeds. Never discover the interesting
            configuration and evaluate on the same episodes -- that error was
            made and caught here at sigma=0.35/B=4.
    Gate 5  only now does observable-state -> allocation become a learning
            problem.

## Two procedural rules that earned their place

- **Execute policies, never sum local proxies.** Run I scored every policy by
  summing local Δ\* along a single H trajectory, so no policy was ever executed.
  Any allocation claim must come from actually running the policy.
- **Gate before build.** The adequacy and allocation gates cost 35 minutes and
  prevented an hour of executor work plus a headline number that would have been
  retracted. Every environment tested *after* a model was built cost days.
