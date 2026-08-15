"""cube_nm_repro — NumPy reproduction of AFABench's CubeNMDataset.

DELIBERATELY NOT called "the AFABench implementation". It is a reproduction, and
it stays named `repro` until it passes the reference-equivalence tests in
`validate()`. This project has voided four results by modelling an environment it
had not executed; a hand port is the same hazard, so it carries a gate.

Ported line-by-line from the released reference at
github.com/Linusaronsson/AFA-Benchmark, afabench/datasets/datasets.py::CubeNMDataset
(MIT). torch is unavailable on this machine (2.5-3 GB against 4.3 GB free at 98%
capacity), so only the generator is reproduced — not the training stack.

WHY THIS DATASET. The label is drawn independently of the context, so the context
features carry ZERO mutual information with the label: I(y; c) = 0. The context is
nonetheless the only thing identifying WHICH of the five blocks carries the label
signal. An action worth nothing now and everything next.

CORRECTION (measured, scripts/cube_nm_myopic.py). An earlier version of this
docstring concluded from I(y; c) = 0 that "a myopic acquirer will never buy the
context". That is FALSE and the exact myopic policy refutes it: it buys the
context at step 2 or 3 in 71% of rows. The zero-information argument holds only
at step 1, from the empty state. Once any block feature has been observed,
I(y; c | x_block) > 0 -- knowing the context is what tells you whether the value
you already hold is signal or noise. Myopia here costs a one-step DELAY, not a
permanent blind spot, and the delay only bites while the budget is tight enough
for one wasted step to matter.

The claim was written from the generative structure without executing a policy
against it. That is the same failure that voided four AgentCE results: reasoning
about an environment instead of running it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

BLOCK_SIZE = 10          # reference: block_size = 10
N_LABELS = 8             # reference: label_shape = [8]


@dataclass(slots=True)
class CubeNMRepro:
    """Reference parameter names and defaults preserved exactly."""

    n_samples: int = 20000
    seed: int = 123
    context_feature_std: float = 0.1
    informative_feature_std: float = 0.1
    non_informative_feature_mean: float = 0.5
    non_informative_feature_std: float = 0.3
    n_contexts: int = 5
    features: np.ndarray = field(init=False)
    labels: np.ndarray = field(init=False)
    context: np.ndarray = field(init=False)

    @property
    def n_features(self) -> int:
        # reference: n_contexts + n_contexts * block_size
        return self.n_contexts + self.n_contexts * BLOCK_SIZE

    @property
    def n_groups(self) -> int:
        """Acquisition groups: the whole context is ONE action, then each block
        feature individually. Reference docstring: 55 features, 51 groups."""
        return 1 + self.n_contexts * BLOCK_SIZE

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        n, k = self.n_samples, self.n_contexts

        context = rng.integers(0, k, size=n)
        onehot = np.eye(k)[context] + rng.normal(0.0, self.context_feature_std, (n, k))
        y_int = rng.integers(0, N_LABELS, size=n)

        # reference: format(i,"03b") then .flip(-1)  -> reversed bit order
        codes = np.array([[int(b) for b in format(i, "03b")][::-1] for i in range(N_LABELS)],
                         dtype=float)

        blocks = rng.normal(self.non_informative_feature_mean,
                            self.non_informative_feature_std,
                            (n, k, BLOCK_SIZE))
        # reference: only the context-selected block carries label information
        for i in range(n):
            ctx, lab = int(context[i]), int(y_int[i])
            idx = [(lab + j) % BLOCK_SIZE for j in range(3)]
            blocks[i, ctx, idx] = codes[lab] + rng.normal(0.0, self.informative_feature_std, 3)

        self.features = np.hstack([onehot, blocks.reshape(n, -1)])
        self.labels = y_int
        self.context = context

    # -- acquisition interface -------------------------------------------------

    def group_columns(self, g: int) -> list[int]:
        """Columns revealed by acquiring group g. Group 0 is the whole context."""
        if g == 0:
            return list(range(self.n_contexts))
        return [self.n_contexts + (g - 1)]

    def block_group_ids(self, ctx: int) -> list[int]:
        """Acquisition groups belonging to the block that context `ctx` selects."""
        start = ctx * BLOCK_SIZE
        return [1 + start + j for j in range(BLOCK_SIZE)]


def validate(ds: CubeNMRepro) -> dict:
    """Reference-equivalence gate. The port is not usable until this passes."""
    checks = {}
    checks["n_features == 55"] = (ds.n_features == 55, ds.n_features)
    checks["n_groups == 51"] = (ds.n_groups == 51, ds.n_groups)
    checks["labels are 8-way"] = (len(np.unique(ds.labels)) == N_LABELS,
                                  len(np.unique(ds.labels)))
    checks["feature matrix shape"] = (ds.features.shape == (ds.n_samples, 55),
                                      ds.features.shape)
    # context must be INDEPENDENT of the label -- this is the whole point
    from sklearn.metrics import mutual_info_score
    mi_ctx = mutual_info_score(ds.context, ds.labels)
    checks["I(context; label) ~ 0"] = (mi_ctx < 0.01, round(float(mi_ctx), 5))
    # the selected block must be informative, a wrong block must not be
    n = min(4000, ds.n_samples)
    sel, wrong = [], []
    for i in range(n):
        c = ds.context[i]
        sel.append(ds.features[i, ds.n_contexts + c * BLOCK_SIZE:
                               ds.n_contexts + (c + 1) * BLOCK_SIZE])
        w = (c + 1) % ds.n_contexts
        wrong.append(ds.features[i, ds.n_contexts + w * BLOCK_SIZE:
                                 ds.n_contexts + (w + 1) * BLOCK_SIZE])
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    y = ds.labels[:n]
    a_sel = cross_val_score(LogisticRegression(max_iter=400), np.array(sel), y, cv=3).mean()
    a_wrong = cross_val_score(LogisticRegression(max_iter=400), np.array(wrong), y, cv=3).mean()
    checks["selected block informative"] = (a_sel > 0.5, round(float(a_sel), 3))
    checks["wrong block ~ chance"] = (a_wrong < 0.25, round(float(a_wrong), 3))
    return checks
