"""Environment 4a — the probe family. Implements PREREGISTRATION-probe-family.md.

Do not read this file as a design document. The design is preregistered in
PREREGISTRATION-probe-family.md rev2 and this module implements it without
deviation. Every parameter here is fixed there.

WHAT IS NEW RELATIVE TO THE GATED FAMILY, which is frozen and untouched.

One acquisition group, `probe`, returning a single scalar

    s ~ Normal(sigma_other, sigma_probe)

and nothing else. Under hypothesis h = (regime r, context c, label y) the
probe's likelihood depends on r ALONE. That is the entire point: the gated
family died because its only regime evidence was also its only label evidence,
so identification was paid for in the currency it was trying to save. Here the
two currencies are structurally separate:

    I(probe ; y | c) = 0
    I(probe ; c)     = 0
    I(probe ; sigma_other) > 0

so the probe cannot improve label accuracy through any path, and if buying it is
ever worth its price, that purchase is metareasoning by construction.

These are claims about code, not about mathematics, and this project has been
burned three times by exactly that distinction. They are gated by
`validate_probe_construction` (preregistered G1/G2a/G2b/G2c) and pinned by
tests, not asserted here.

IMPLEMENTATION NOTE. The probe gets its own quadrature grid. Block columns live
on [-2, 3] with 241 nodes, which is ~8 nodes per standard deviation at the block
scale; at sigma_probe = 0.05 that same grid would give 3 nodes per sd and the
expected-entropy integral would be visibly wrong. The probe grid is separate and
fine, which costs almost nothing because the probe is a single column.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from governor.envs.gated_family import (
    CODE_BITS,
    N_LABELS,
    REGIME_GRID,
    REGIME_PRIOR,
    GateConfig,
    _codes,
)

_LOG2PI = float(np.log(2.0 * np.pi))


@dataclass(frozen=True, slots=True)
class ProbeConfig:
    """GateConfig plus the probe. Never exposed to a policy."""

    base: GateConfig = field(default_factory=GateConfig)
    sigma_probe: float = 0.05        # preregistered: {0.05, 0.15}
    probe_cost: float = 0.25         # preregistered operating point, no sweep

    @property
    def n_contexts(self) -> int:
        return self.base.n_contexts

    @property
    def block_size(self) -> int:
        return self.base.block_size

    @property
    def n_features(self) -> int:
        return self.base.n_features + 1          # + the probe column

    @property
    def probe_col(self) -> int:
        return self.base.n_features              # appended last

    @property
    def n_groups(self) -> int:
        return self.base.n_groups + 1            # + the probe group

    @property
    def probe_group(self) -> int:
        return self.base.n_groups                # appended last


@dataclass(slots=True)
class ProbeTask:
    """Gated-family data with one extra column carrying only regime evidence."""

    cfg: ProbeConfig
    n_samples: int = 3000
    seed: int = 0
    features: np.ndarray = field(init=False)
    labels: np.ndarray = field(init=False)
    context: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        b = self.cfg.base
        rng = np.random.default_rng(self.seed)
        n, K, M = self.n_samples, b.n_contexts, b.block_size

        ctx = rng.integers(0, K, size=n)
        gate = np.eye(K)[ctx] + rng.normal(0.0, b.sigma_gate, (n, K))
        y = rng.integers(0, N_LABELS, size=n)
        codes = _codes()

        blocks = rng.normal(b.filler_mean, b.filler_std, (n, K, M))
        for i in range(n):
            lab, cc = int(y[i]), int(ctx[i])
            idx = [(lab + j) % M for j in range(CODE_BITS)]
            for bi in range(K):
                sd = b.sigma_sig if bi == cc else b.sigma_other
                blocks[i, bi, idx] = codes[lab] + rng.normal(0.0, sd, CODE_BITS)

        # THE PROBE. Drawn from the regime alone -- it does not see y or c, so
        # no label information can enter through this column by construction.
        probe = rng.normal(b.sigma_other, self.cfg.sigma_probe, (n, 1))

        self.features = np.hstack([gate, blocks.reshape(n, -1), probe])
        self.labels = y
        self.context = ctx

    def group_columns(self, g: int) -> list[int]:
        if g == self.cfg.probe_group:
            return [self.cfg.probe_col]
        if g == 0:
            return list(range(self.cfg.n_contexts))
        return [self.cfg.n_contexts + (g - 1)]

    def group_cost(self, g: int) -> float:
        if g == self.cfg.probe_group:
            return self.cfg.probe_cost
        return self.cfg.base.gate_cost if g == 0 else self.cfg.base.block_cost


class ProbeBayes:
    """Exact posterior over (regime, context, label) with a regime-only probe."""

    def __init__(self, task: ProbeTask, *, regimes: tuple[float, ...] | None = None,
                 prior: tuple[float, ...] | None = None,
                 n_gate_nodes: int = 192, grid_nodes: int = 241,
                 probe_nodes: int = 641) -> None:
        c = task.cfg
        b = c.base
        self.task, self.cfg = task, c
        self.regimes = tuple(regimes) if regimes is not None else (b.sigma_other,)
        self.R = len(self.regimes)
        self.knows_regime = regimes is None
        self.K, self.M = b.n_contexts, b.block_size
        self.H = self.R * self.K * N_LABELS
        self.nf = c.n_features
        codes = _codes()

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
                    sd[h, : self.K] = b.sigma_gate
                    mu[h, self.K:c.probe_col] = b.filler_mean
                    sd[h, self.K:c.probe_col] = b.filler_std
                    for bi in range(self.K):
                        base = self.K + bi * self.M
                        s = b.sigma_sig if bi == ctx else s_other
                        for j in range(CODE_BITS):
                            col = base + (y + j) % self.M
                            mu[h, col] = codes[y][j]
                            sd[h, col] = s
                    # the probe: a function of the REGIME index only. Note it
                    # does not read ctx or y at all -- that is the decoupling,
                    # visible in one line.
                    mu[h, c.probe_col] = s_other
                    sd[h, c.probe_col] = c.sigma_probe
        self.MU, self.SD, self.LOGSD = mu, sd, np.log(sd)

        self.group_cols = [task.group_columns(g) for g in range(c.n_groups)]
        self.cost = np.array([task.group_cost(g) for g in range(c.n_groups)])
        self.n_groups = c.n_groups
        self.probe_group = c.probe_group

        # block-column quadrature
        self._grid = np.linspace(-2.0, 3.0, grid_nodes)
        bc = np.arange(self.K, c.probe_col)
        self._col_slot = {int(x): i for i, x in enumerate(bc)}
        d = (self._grid[None, None, :] - mu[:, bc][:, :, None]) / sd[:, bc][:, :, None]
        self._pdf = np.exp(-0.5 * d * d) / sd[:, bc][:, :, None]

        # SEPARATE, FINER grid for the probe. At sigma_probe=0.05 the block grid
        # would give ~3 nodes per standard deviation and the expected-entropy
        # integral would be materially wrong.
        lo = min(self.regimes) - 6 * c.sigma_probe
        hi = max(self.regimes) + 6 * c.sigma_probe
        self._pgrid = np.linspace(lo, hi, probe_nodes)
        dp = (self._pgrid[None, :] - mu[:, c.probe_col][:, None]) \
            / c.sigma_probe
        self._ppdf = np.exp(-0.5 * dp * dp) / c.sigma_probe

        self._gz = np.random.default_rng(20260816).standard_normal(
            (n_gate_nodes, self.K))

    # -- inference -------------------------------------------------------------

    def prior_logL(self) -> np.ndarray:
        return self.log_prior.copy()

    def loglik_cols(self, x: np.ndarray, cols: list[int]) -> np.ndarray:
        if not cols:
            return np.zeros(self.H)
        d = (x[cols][None, :] - self.MU[:, cols]) / self.SD[:, cols]
        return (-0.5 * d * d - self.LOGSD[:, cols] - 0.5 * _LOG2PI).sum(axis=1)

    def _norm(self, logL):
        p = np.exp(logL - logL.max())
        return p / p.sum()

    def label_posterior(self, logL):
        return self._norm(logL).reshape(self.R, self.K, N_LABELS).sum(axis=(0, 1))

    def regime_posterior(self, logL):
        return self._norm(logL).reshape(self.R, self.K, N_LABELS).sum(axis=(1, 2))

    def context_posterior(self, logL):
        return self._norm(logL).reshape(self.R, self.K, N_LABELS).sum(axis=(0, 2))

    def predict(self, x, cols):
        return int(np.argmax(self.label_posterior(
            self.prior_logL() + self.loglik_cols(x, cols))))

    def _marg(self, p, target):
        s = p.reshape(*p.shape[:-1], self.R, self.K, N_LABELS)
        return s.sum(axis=(-2, -1)) if target == "regime" else s.sum(axis=(-3, -2))

    def _ent(self, logL, target="label"):
        m = logL.max(axis=-1, keepdims=True)
        p = np.exp(logL - m)
        p /= p.sum(axis=-1, keepdims=True)
        q = self._marg(p, target)
        return -(q * np.log(np.maximum(q, 1e-300))).sum(axis=-1)

    # -- acquisition -----------------------------------------------------------

    def gains(self, logL, available, target="label"):
        post = self._norm(logL)
        h_now = float(self._ent(logL[None, :], target)[0])
        out: dict[int, float] = {}

        single = [g for g in available if g not in (0, self.probe_group)]
        if single:
            slots = np.array([self._col_slot[self.group_cols[g][0]] for g in single])
            joint = post[:, None, None] * self._pdf[:, slots, :]
            px = joint.sum(axis=0)
            w = px / px.sum(axis=1, keepdims=True)
            cond = joint / np.maximum(px[None, :, :], 1e-300)
            sh = cond.reshape(self.R, self.K, N_LABELS, len(single), -1)
            pm = sh.sum(axis=(1, 2)) if target == "regime" else sh.sum(axis=(0, 1))
            e = -(pm * np.log(np.maximum(pm, 1e-300))).sum(axis=0)
            # pm is (marginal, G, T) for either target, so e is (G, T)
            for i, g in enumerate(single):
                out[g] = h_now - float((w[i] * e[i]).sum())

        if self.probe_group in available:
            joint = post[:, None] * self._ppdf              # (H, T)
            px = joint.sum(axis=0)
            w = px / px.sum()
            cond = joint / np.maximum(px[None, :], 1e-300)
            sh = cond.reshape(self.R, self.K, N_LABELS, -1)
            pm = sh.sum(axis=(1, 2)) if target == "regime" else sh.sum(axis=(0, 1))
            e = -(pm * np.log(np.maximum(pm, 1e-300))).sum(axis=0)
            out[self.probe_group] = h_now - float((w * e).sum())

        if 0 in available:
            cols = self.group_cols[0]
            pc = post.reshape(self.R, self.K, N_LABELS).sum(axis=(0, 2))
            tot = 0.0
            for ctx in range(self.K):
                if pc[ctx] < 1e-12:
                    continue
                xg = np.eye(self.K)[ctx][None, :] + self.cfg.base.sigma_gate * self._gz
                d = (xg[:, None, :] - self.MU[None, :, cols]) / self.SD[None, :, cols]
                ll = (-0.5 * d * d - self.LOGSD[None, :, cols]).sum(axis=-1)
                tot += pc[ctx] * float(self._ent(logL[None, :] + ll, target).mean())
            out[0] = h_now - tot
        return out

    def myopic_step(self, logL, available, remaining):
        afford = [g for g in available if self.cost[g] <= remaining + 1e-9]
        if not afford:
            return None
        g = self.gains(logL, afford)
        return max(afford, key=lambda a: g[a] / self.cost[a])


def OracleProbeBayes(task, **kw):  # noqa: N802
    return ProbeBayes(task, regimes=None, **kw)


def ObservableProbeBayes(task, *, regimes=REGIME_GRID, **kw):  # noqa: N802
    return ProbeBayes(task, regimes=regimes, prior=REGIME_PRIOR, **kw)


def make_config(sigma_other: float, gate_cost: float, sigma_probe: float,
                probe_cost: float = 0.25) -> ProbeConfig:
    return ProbeConfig(
        base=replace(GateConfig(), sigma_other=sigma_other, gate_cost=gate_cost),
        sigma_probe=sigma_probe, probe_cost=probe_cost)
