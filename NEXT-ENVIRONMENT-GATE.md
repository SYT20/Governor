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

    M1  allocation diversity
        number of distinct optimal schedules, and the fraction of episodes
        whose optimal schedule differs from the BEST CONSTANT SCHEDULE

    M2  adaptive allocation value
        U(oracle-adaptive) - U(best-constant-schedule)  on held-out episodes

**Reject the environment if the best constant schedule captures essentially all
available value.** That single comparison is what Env 5 lacked and it is cheap
— it needs no learned model, no executor, and no Governor.

## Two procedural rules that earned their place

- **Execute policies, never sum local proxies.** Run I scored every policy by
  summing local Δ\* along a single H trajectory, so no policy was ever executed.
  Any allocation claim must come from actually running the policy.
- **Gate before build.** The adequacy and allocation gates cost 35 minutes and
  prevented an hour of executor work plus a headline number that would have been
  retracted. Every environment tested *after* a model was built cost days.
