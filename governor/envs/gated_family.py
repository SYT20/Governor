"""A task family where deliberation is sometimes worth it, sometimes harmful.

WHY THIS EXISTS. Fixed-structure CUBE-NM was measured to be degenerate for the
Governor question: the whole usable override signal was explained by budget and
progress, and the apparent cognitive residual (+0.035) vanished under an honest
progress control (+0.023, CI [-0.006, +0.051]). The environment never required
the controller to infer anything about the task, because there was only one task.

THE DESIGN CONSTRAINT THAT DRIVES EVERYTHING. Two tasks must be able to look
IDENTICAL in every coarse regime variable -- same budget, same horizon, same
group count, same feature dimensionality, same initial uncertainty -- and still
demand opposite metacognitive decisions. If that is not true, a lookup table
solves the benchmark again and the experiment measures nothing. It is enforced
here by construction: every configuration below produces the same 1 + K*M
acquisition groups over the same feature space, and the ONLY thing that differs
is a noise parameter invisible until the agent starts observing.

THE MECHANISM. One `gate` group reveals the latent context c (noisy one-hot).
Context c designates a primary block. Every block carries the label code at
positions {y, y+1, y+2}, but with block-dependent noise:

    primary block  (b == c) : sd = sigma_sig     tight, informative
    other blocks   (b != c) : sd = sigma_other   varies by configuration

`sigma_other` is the single parameter that decides whether deliberation pays,
and it interpolates continuously between three regimes:

    sigma_other >> sigma_sig   only the primary block informs
                               -> the gate is ESSENTIAL, Delta_meta > 0
    sigma_other ~= sigma_sig   every block informs equally, c is irrelevant
                               -> the gate is a WASTED slot, Delta_meta < 0
    sigma_other  > sigma_sig   others inform weakly
                               -> marginal, Delta_meta ~ 0

The harmful case is the important one. Without tasks where strategic reasoning
LOSES, a controller scores well by always deliberating and we cannot distinguish
that from competence.

OBSERVABILITY CONTRACT -- read this before using any scorer here.

An earlier version of this docstring claimed "nothing about the configuration is
observable to a policy". That was FALSE about the code beneath it. The scorer
built its likelihood from `cfg.sigma_other`, so it knew the regime outright:
asked for the best first acquisition from the EMPTY state, it returned gain
0.5537 under sigma_other=0.10 and 0.2939 under 1.50. A policy that has observed
nothing cannot produce two different numbers. It was reading the answer.

The class therefore takes an explicit `regimes` argument and there are two
legitimate constructions:

    OracleBayes(task)          regimes = (true sigma_other,)
                               Knows the regime. Valid ONLY for upper bounds,
                               environment validation, and generating teacher
                               labels. Never a stand-in for an agent.

    ObservableBayes(task)      regimes = REGIME_GRID
                               Carries a joint posterior over
                               (sigma_other, context, label) and marginalises.
                               Sees only its own observations. This is the one
                               a policy claim may be made about.

That upgrade is not merely a correctness patch -- it is what makes the family
interesting. The observable agent now faces two coupled unknowns: WHICH INSTANCE
it is solving (context, label) and WHAT KIND OF TASK it is in (sigma_other). The
metacognitive question "is deliberation worth paying for here?" is unanswerable
until the second one is partly resolved, and resolving it costs budget.

`sigma_gate` is assumed known; keeping exactly one unknown regime parameter
keeps the hypothesis space at R*K*8 and the posterior exact.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

_LOG2PI = float(np.log(2.0 * np.pi))
N_LABELS = 8
CODE_BITS = 3


@dataclass(frozen=True, slots=True)
class GateConfig:
    """One point in the factorial task family. Never exposed to the policy."""

    n_contexts: int = 5
    block_size: int = 10
    sigma_sig: float = 0.10          # noise on the primary block's code
    sigma_other: float = 1.50        # noise on every other block's code
    sigma_gate: float = 0.10         # noise on the gate's one-hot
    gate_cost: float = 1.0           # cost of acquiring the gate
    block_cost: float = 1.0          # cost of one block feature
    filler_mean: float = 0.50
    filler_std: float = 0.30

    @property
    def n_features(self) -> int:
        return self.n_contexts + self.n_contexts * self.block_size

    @property
    def n_groups(self) -> int:
        return 1 + self.n_contexts * self.block_size

    def key(self) -> str:
        return (f"K{self.n_contexts}_so{self.sigma_other:g}_"
                f"sg{self.sigma_gate:g}_gc{self.gate_cost:g}")


def _codes() -> np.ndarray:
    return np.array([[int(b) for b in format(i, "03b")][::-1]
                     for i in range(N_LABELS)], dtype=float)


@dataclass(slots=True)
class GatedTask:
    """Dataset for one configuration."""

    cfg: GateConfig
    n_samples: int = 4000
    seed: int = 0
    features: np.ndarray = field(init=False)
    labels: np.ndarray = field(init=False)
    context: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        c = self.cfg
        rng = np.random.default_rng(self.seed)
        n, K, M = self.n_samples, c.n_contexts, c.block_size

        ctx = rng.integers(0, K, size=n)
        gate = np.eye(K)[ctx] + rng.normal(0.0, c.sigma_gate, (n, K))
        y = rng.integers(0, N_LABELS, size=n)
        codes = _codes()

        blocks = rng.normal(c.filler_mean, c.filler_std, (n, K, M))
        for i in range(n):
            lab, cc = int(y[i]), int(ctx[i])
            idx = [(lab + j) % M for j in range(CODE_BITS)]
            for b in range(K):
                sd = c.sigma_sig if b == cc else c.sigma_other
                blocks[i, b, idx] = codes[lab] + rng.normal(0.0, sd, CODE_BITS)

        self.features = np.hstack([gate, blocks.reshape(n, -1)])
        self.labels = y
        self.context = ctx

    def group_columns(self, g: int) -> list[int]:
        if g == 0:
            return list(range(self.cfg.n_contexts))
        return [self.cfg.n_contexts + (g - 1)]

    def group_cost(self, g: int) -> float:
        return self.cfg.gate_cost if g == 0 else self.cfg.block_cost


REGIME_GRID: tuple[float, ...] = (0.10, 0.20, 0.35, 0.60, 1.50)
"""Candidate sigma_other values an observable agent entertains.

Matches the grid the family is generated from, so the observable posterior is
well-specified rather than misspecified. Misspecification is a separate research
question and mixing it in here would confound the metacognition result.
"""

REGIME_PRIOR: tuple[float, ...] = (0.2, 0.2, 0.2, 0.2, 0.2)
"""PREREGISTERED prior over REGIME_GRID. Uniform, fixed before any Phase 2
measurement, and deliberately NOT inherited from the frequency with which each
regime appears in whatever configuration grid an experiment happens to sweep.

Inheriting it would smuggle knowledge of the benchmark's construction into the
"observable" agent: a prior tuned to the test distribution is privileged
information wearing a Bayesian costume. Tuning it after seeing probe results
would be the same gate-shopping that made the ECE gate meaningless earlier in
this project. It is asserted by test, not by comment -- see
test_regime_posterior_starts_at_the_preregistered_prior.
"""


class GatedBayes:
    """Exact posterior over (regime, context, label), cost-aware myopic play.

    Generalises the CUBE-NM scorer. Two differences that matter:

    COST AWARENESS. Groups no longer cost the same, so the myopic rule maximises
    expected information gain PER UNIT COST, not raw gain. Ranking by raw gain
    under non-uniform costs is not myopic-optimal, it is just wrong, and it would
    hand the strategic arm a free win whenever the gate happened to be expensive.

    THE GATE IS NO LONGER NEAR-DETERMINISTIC. CUBE-NM's context observation had
    sigma=0.1 against unit separation, so acquiring it identified c with error
    ~1e-12 and the expectation collapsed to K terms. Here sigma_gate varies, so
    that collapse is invalid. Instead the expectation is taken EXACTLY over the
    K possible true contexts and by fixed-table quadrature over the K noise
    draws -- deterministic, reproducible, and not a sampling estimate whose
    convergence would have to be argued.
    """

    def __init__(self, task: GatedTask, *, regimes: tuple[float, ...] | None = None,
                 prior: tuple[float, ...] | None = None,
                 n_gate_nodes: int = 192, grid_nodes: int = 241) -> None:
        c = task.cfg
        self.task, self.cfg = task, c
        # regimes=None means ORACLE: collapse the regime axis onto the truth.
        self.regimes = tuple(regimes) if regimes is not None else (c.sigma_other,)
        self.R = len(self.regimes)
        self.knows_regime = regimes is None
        self.K, self.M = c.n_contexts, c.block_size
        self.H = self.R * self.K * N_LABELS
        self.nf = c.n_features
        codes = _codes()

        # Carried as a log-prior over hypotheses so that the EMPTY state is the
        # preregistered prior rather than an implicit uniform. Context and label
        # are uniform by construction of the generator.
        p = np.asarray(prior if prior is not None else [1.0 / self.R] * self.R,
                       dtype=float)
        if p.shape != (self.R,):
            raise ValueError(f"prior must have {self.R} entries, got {p.shape}")
        self.log_prior = np.repeat(np.log(p / p.sum()), self.K * N_LABELS)

        mu = np.empty((self.H, self.nf))
        sd = np.empty((self.H, self.nf))
        for r, s_other in enumerate(self.regimes):
            for ctx in range(self.K):
                for y in range(N_LABELS):
                    h = (r * self.K + ctx) * N_LABELS + y
                    mu[h, : self.K] = 0.0
                    mu[h, ctx] = 1.0
                    sd[h, : self.K] = c.sigma_gate
                    mu[h, self.K:] = c.filler_mean
                    sd[h, self.K:] = c.filler_std
                    for b in range(self.K):
                        base = self.K + b * self.M
                        s = c.sigma_sig if b == ctx else s_other
                        for j in range(CODE_BITS):
                            col = base + (y + j) % self.M
                            mu[h, col] = codes[y][j]
                            sd[h, col] = s
        self.MU, self.SD, self.LOGSD = mu, sd, np.log(sd)

        self.group_cols = [task.group_columns(g) for g in range(c.n_groups)]
        self.cost = np.array([task.group_cost(g) for g in range(c.n_groups)])
        self.n_groups = c.n_groups

        # quadrature table for 1-D block observations
        self._grid = np.linspace(-2.0, 3.0, grid_nodes)
        bc = np.arange(self.K, self.nf)
        self._col_slot = {int(x): i for i, x in enumerate(bc)}
        d = (self._grid[None, None, :] - mu[:, bc][:, :, None]) / sd[:, bc][:, :, None]
        self._pdf = np.exp(-0.5 * d * d) / sd[:, bc][:, :, None]

        # fixed deterministic noise table for the K-dimensional gate observation
        self._gz = np.random.default_rng(20260816).standard_normal(
            (n_gate_nodes, self.K))

    # -- inference -------------------------------------------------------------

    def loglik_cols(self, x: np.ndarray, cols: list[int]) -> np.ndarray:
        if not cols:
            return np.zeros(self.H)
        d = (x[cols][None, :] - self.MU[:, cols]) / self.SD[:, cols]
        return (-0.5 * d * d - self.LOGSD[:, cols] - 0.5 * _LOG2PI).sum(axis=1)

    def prior_logL(self) -> np.ndarray:
        """The empty state: the preregistered prior, not a flat vector."""
        return self.log_prior.copy()

    def _norm(self, logL: np.ndarray) -> np.ndarray:
        p = np.exp(logL - logL.max())
        return p / p.sum()

    def label_posterior(self, logL: np.ndarray) -> np.ndarray:
        """P(y | obs), marginalising out BOTH the regime and the context."""
        return self._norm(logL).reshape(self.R, self.K, N_LABELS).sum(axis=(0, 1))

    def regime_posterior(self, logL: np.ndarray) -> np.ndarray:
        """P(sigma_other | obs) -- the agent's belief about WHAT KIND of task.

        This is the metacognitive belief. It is what makes "should I pay to
        deliberate?" answerable at all, and under the oracle construction it is
        degenerate at the truth, which is exactly why the oracle must never be
        used as a stand-in for a policy.
        """
        return self._norm(logL).reshape(self.R, self.K, N_LABELS).sum(axis=(1, 2))

    def context_posterior(self, logL: np.ndarray) -> np.ndarray:
        return self._norm(logL).reshape(self.R, self.K, N_LABELS).sum(axis=(0, 2))

    def predict(self, x: np.ndarray, cols: list[int]) -> int:
        return int(np.argmax(self.label_posterior(
            self.prior_logL() + self.loglik_cols(x, cols))))

    def _ent_y(self, logL: np.ndarray) -> np.ndarray:
        m = logL.max(axis=-1, keepdims=True)
        p = np.exp(logL - m)
        p /= p.sum(axis=-1, keepdims=True)
        py = p.reshape(*p.shape[:-1], self.R, self.K, N_LABELS).sum(axis=(-3, -2))
        return -(py * np.log(np.maximum(py, 1e-300))).sum(axis=-1)

    # -- acquisition -----------------------------------------------------------

    def gains(self, logL: np.ndarray, available: list[int]) -> dict[int, float]:
        """Expected reduction in H(y) for each candidate group."""
        post = np.exp(logL - logL.max())
        post /= post.sum()
        h_now = float(self._ent_y(logL[None, :])[0])
        out: dict[int, float] = {}

        single = [g for g in available if g != 0]
        if single:
            slots = np.array([self._col_slot[self.group_cols[g][0]] for g in single])
            joint = post[:, None, None] * self._pdf[:, slots, :]
            px = joint.sum(axis=0)
            w = px / px.sum(axis=1, keepdims=True)
            cond = joint / np.maximum(px[None, :, :], 1e-300)
            py = cond.reshape(self.R, self.K, N_LABELS, len(single), -1
                              ).sum(axis=(0, 1))
            ent = -(py * np.log(np.maximum(py, 1e-300))).sum(axis=0)
            val = (w * ent).sum(axis=1)
            for i, g in enumerate(single):
                out[g] = h_now - float(val[i])

        if 0 in available:
            # exact over the K true contexts, quadrature over the K noise draws
            cols = self.group_cols[0]
            pc = post.reshape(self.R, self.K, N_LABELS).sum(axis=(0, 2))
            tot = 0.0
            for ctx in range(self.K):
                if pc[ctx] < 1e-12:
                    continue
                x = np.eye(self.K)[ctx][None, :] + self.cfg.sigma_gate * self._gz
                d = (x[:, None, :] - self.MU[None, :, cols]) / self.SD[None, :, cols]
                ll = (-0.5 * d * d - self.LOGSD[None, :, cols]).sum(axis=-1)
                tot += pc[ctx] * float(self._ent_y(logL[None, :] + ll).mean())
            out[0] = h_now - tot
        return out

    def myopic_step(self, logL: np.ndarray, available: list[int],
                    remaining: float) -> int | None:
        """Cost-aware myopic choice: best expected gain PER UNIT COST.

        Ranking by raw gain would be incorrect under non-uniform costs and would
        gift the strategic arm a win whenever the gate happened to be expensive.
        """
        afford = [g for g in available if self.cost[g] <= remaining + 1e-9]
        if not afford:
            return None
        g = self.gains(logL, afford)
        return max(afford, key=lambda a: g[a] / self.cost[a])

    def run(self, x: np.ndarray, budget: float, *,
            forced_first: int | None = None) -> tuple[list[int], int, float]:
        """Play to exhaustion of the budget; return (groups, prediction, spend)."""
        logL = self.prior_logL()
        available = list(range(self.n_groups))
        got: list[int] = []
        spent = 0.0
        if forced_first is not None and self.cost[forced_first] <= budget + 1e-9:
            available.remove(forced_first)
            got.append(forced_first)
            spent += float(self.cost[forced_first])
            logL = logL + self.loglik_cols(x, self.group_cols[forced_first])
        while True:
            g = self.myopic_step(logL, available, budget - spent)
            if g is None:
                break
            available.remove(g)
            got.append(g)
            spent += float(self.cost[g])
            logL = logL + self.loglik_cols(x, self.group_cols[g])
        return got, int(np.argmax(self.label_posterior(logL))), spent


def OracleBayes(task: GatedTask, **kw) -> GatedBayes:  # noqa: N802
    """Knows the true regime. Upper bounds and teacher labels ONLY."""
    return GatedBayes(task, regimes=None, **kw)


def ObservableBayes(task: GatedTask, *, regimes=REGIME_GRID, **kw) -> GatedBayes:  # noqa: N802
    """Sees only its own observations; infers the regime. Policy claims use this."""
    return GatedBayes(task, regimes=regimes, **kw)


def delta_meta(cfg: GateConfig, budget: float, *, n: int = 400,
               seed: int = 0, observable: bool = True) -> dict:
    """Measured value of forcing the gate first, at one (config, budget).

    Delta_meta = V(gate first, then myopic) - V(pure myopic), in realised
    accuracy on n instances. This is the quantity the Governor must learn to
    predict, and the quantity whose SIGN must vary across the family for the
    switching problem to be non-trivial.
    """
    task = GatedTask(cfg=cfg, n_samples=n, seed=seed)
    bayes = ObservableBayes(task) if observable else OracleBayes(task)
    hit_m = hit_s = 0
    gate_used = 0
    for i in range(n):
        x, y = task.features[i], int(task.labels[i])
        gm, pm, _ = bayes.run(x, budget)
        _, ps, _ = bayes.run(x, budget, forced_first=0)
        hit_m += int(pm == y)
        hit_s += int(ps == y)
        gate_used += int(0 in gm)
    return {
        "myopic": hit_m / n,
        "strategic": hit_s / n,
        "delta_meta": (hit_s - hit_m) / n,
        "myopic_buys_gate": gate_used / n,
        "observable": observable,
    }
