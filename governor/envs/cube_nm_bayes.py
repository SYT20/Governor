"""Exact Bayesian inference and exact myopic acquisition for CUBE-NM.

WHY THIS EXISTS. The previous gate compared context-first against
`greedy_predictive_order`, which computes ONE feature ordering per dataset on
training data and replays it on every test row. That is a static ordering, not a
myopic policy: it never conditions on what the current instance has already
revealed. Beating it does not establish "non-myopic beats myopic" -- only
"non-myopic beats a fixed acquisition schedule", which is a much weaker claim.

The fix could have been a model-based adaptive greedy (impute, refit, score).
That would remain attackable: any loss could be blamed on a weak density model
rather than on myopia. CUBE-NM is a fully specified generative process, so
instead we hand the myopic policy the TRUE posterior. It then plays the
Bayes-optimal one-step-lookahead strategy -- the strongest myopic policy that
can exist. If it still loses, the deficit is attributable to the decision
horizon and to nothing else.

Generative model, read directly off cube_nm_repro.__post_init__:

    c ~ U{0..K-1},  y ~ U{0..7}                          (independent)
    context column j  ~ N(1[j == c], context_feature_std)
    block column (b,m) ~ N(0.5, 0.3)                     unless b == c
    block column (c,m) ~ N(codes[y][m-y], 0.1)           for m in {y, y+1, y+2}

y <= 7 and BLOCK_SIZE == 10, so the (y+j) % 10 wrap in the reference never
fires; indices are literally y, y+1, y+2.

Every observation is conditionally independent given (c, y), so the posterior
over the K*8 = 40 joint hypotheses is a product of Gaussian densities and is
exact, not approximate. `validate_likelihood` is the gate that proves the
parameterisation here matches the data generator -- a mis-derived likelihood
would silently hand the myopic policy a broken model and manufacture the very
result this file exists to attack.
"""

from __future__ import annotations

import numpy as np

from governor.envs.cube_nm_repro import BLOCK_SIZE, N_LABELS, CubeNMRepro

_LOG2PI = float(np.log(2.0 * np.pi))


class CubeNMBayes:
    """Exact posterior over (context, label) and exact myopic acquisition."""

    def __init__(self, ds: CubeNMRepro) -> None:
        self.ds = ds
        self.K = ds.n_contexts
        self.H = self.K * N_LABELS                    # 40 joint hypotheses
        self.n_features = ds.n_features

        codes = np.array(
            [[int(b) for b in format(i, "03b")][::-1] for i in range(N_LABELS)],
            dtype=float,
        )

        mu = np.empty((self.H, self.n_features))
        sd = np.empty((self.H, self.n_features))
        for c in range(self.K):
            for y in range(N_LABELS):
                h = c * N_LABELS + y
                # context one-hot block
                mu[h, : self.K] = 0.0
                mu[h, c] = 1.0
                sd[h, : self.K] = ds.context_feature_std
                # every block feature is non-informative by default ...
                mu[h, self.K:] = ds.non_informative_feature_mean
                sd[h, self.K:] = ds.non_informative_feature_std
                # ... except the three carrying the code, in block c only
                base = self.K + c * BLOCK_SIZE
                for j in range(3):
                    col = base + (y + j) % BLOCK_SIZE
                    mu[h, col] = codes[y][j]
                    sd[h, col] = ds.informative_feature_std

        self.MU = mu
        self.SD = sd
        self.LOGSD = np.log(sd)
        # column list per acquisition group, and the reverse map
        self.group_cols = [ds.group_columns(g) for g in range(ds.n_groups)]
        self.n_groups = ds.n_groups

        # -- exact-quadrature tables, built once per dataset --------------------
        # Every block column is a 1-D observation, so E_x[H(y | x_O, x)] is a
        # ONE-dimensional integral and can be evaluated by quadrature instead of
        # sampled. MU in {0, 0.5, 1} and SD in {0.1, 0.3} bound the support well
        # inside [-1.5, 2.5]; 321 nodes give ~8 per standard deviation at the
        # tightest scale. The density table depends only on (h, column), never on
        # the observation, so it is precomputed here and reused at every step of
        # every instance.
        self._grid = np.linspace(-1.5, 2.5, 321)
        bc = np.arange(self.K, self.n_features)          # block columns
        self._block_cols = bc
        self._col_slot = {int(c): i for i, c in enumerate(bc)}
        d = (self._grid[None, None, :] - mu[:, bc][:, :, None]) / sd[:, bc][:, :, None]
        self._pdf = np.exp(-0.5 * d * d) / sd[:, bc][:, :, None]   # (H, ncols, T)

    # -- inference -------------------------------------------------------------

    def loglik_cols(self, x: np.ndarray, cols: list[int]) -> np.ndarray:
        """log P(x[cols] | h) for every hypothesis h. Shape (H,)."""
        if not cols:
            return np.zeros(self.H)
        d = (x[cols][None, :] - self.MU[:, cols]) / self.SD[:, cols]
        return (-0.5 * d * d - self.LOGSD[:, cols] - 0.5 * _LOG2PI).sum(axis=1)

    def label_posterior(self, logL: np.ndarray) -> np.ndarray:
        """Marginal P(y | observations) from a joint log-likelihood vector."""
        p = np.exp(logL - logL.max())
        p /= p.sum()
        return p.reshape(self.K, N_LABELS).sum(axis=0)

    def predict(self, x: np.ndarray, cols: list[int]) -> int:
        return int(np.argmax(self.label_posterior(self.loglik_cols(x, cols))))

    # -- myopic acquisition ----------------------------------------------------

    def _entropy_y(self, logL: np.ndarray) -> np.ndarray:
        """Shannon entropy of the y-marginal, over a leading batch of shape (...)."""
        m = logL.max(axis=-1, keepdims=True)
        p = np.exp(logL - m)
        p /= p.sum(axis=-1, keepdims=True)
        py = p.reshape(*p.shape[:-1], self.K, N_LABELS).sum(axis=-2)
        return -(py * np.where(py > 0, np.log(np.maximum(py, 1e-300)), 0.0)).sum(axis=-1)

    def myopic_step_exact(self, logL: np.ndarray, available: list[int]) -> int:
        """Bayes-optimal one-step lookahead with NO Monte Carlo anywhere.

            a* = argmax_a  -E_{x_a ~ P(x_a | x_O)}[ H(y | x_O, x_a) ]

        The sampled version of this had to be abandoned: accuracy at budget 4
        rose 0.428 -> 0.471 -> 0.541 as n_mc went 8 -> 32 -> 128, so the myopic
        arm was losing partly to sampling noise. A deficit that shrinks when you
        add samples is an estimator artefact, and attributing it to the planning
        horizon would have been wrong. Both expectations are closed enough to
        evaluate exactly, so no convergence argument is needed at all.

        Block columns: one observation each, so the expectation is a 1-D
        integral over the posterior predictive -- quadrature on the precomputed
        density table.

        Context group: sigma = 0.1 against unit separation, so the observation
        identifies the context with error probability Phi(-1/(0.1*sqrt(2))) ~
        8e-13. The expectation therefore collapses to a 5-term sum over
        P(c | x_O), with the posterior restricted to each c in turn. Exact for
        every practical purpose, and `check_context_score_exactness` verifies it
        against a high-sample estimate rather than asserting it.
        """
        post = np.exp(logL - logL.max())
        post /= post.sum()
        scores: dict[int, float] = {}

        single = [g for g in available if g != 0]
        if single:
            slots = np.array([self._col_slot[self.group_cols[g][0]] for g in single])
            joint = post[:, None, None] * self._pdf[:, slots, :]       # (H, G, T)
            px = joint.sum(axis=0)                                     # (G, T)
            w = px / px.sum(axis=1, keepdims=True)
            cond = joint / np.maximum(px[None, :, :], 1e-300)
            py = cond.reshape(self.K, N_LABELS, len(single), -1).sum(axis=0)
            ent = -(py * np.log(np.maximum(py, 1e-300))).sum(axis=0)   # (G, T)
            val = -(w * ent).sum(axis=1)
            for i, g in enumerate(single):
                scores[g] = float(val[i])

        if 0 in available:
            P = post.reshape(self.K, N_LABELS)
            pc = P.sum(axis=1)
            pyc = P / np.maximum(pc[:, None], 1e-300)
            hc = -(pyc * np.log(np.maximum(pyc, 1e-300))).sum(axis=1)
            scores[0] = -float((pc * hc).sum())

        return max(scores, key=lambda g: scores[g])

    def myopic_step(
        self, logL: np.ndarray, available: list[int], rng: np.random.Generator, n_mc: int
    ) -> int:
        """Bayes-optimal ONE-step-lookahead choice among `available` groups.

        Maximises expected immediate information gain about y,

            a* = argmax_a  H(y | x_O) - E_{x_a ~ P(x_a | x_O)}[ H(y | x_O, x_a) ]

        H(y | x_O) is constant across candidates, so only the second term is
        computed. The expectation is Monte Carlo over the CURRENT posterior:
        draw h ~ P(h | x_O), then x_a ~ P(x_a | h). Correct by construction --
        that composition is exactly the posterior predictive.

        The SAME hypothesis draws and the SAME standard-normal draws are shared
        across every candidate. Common random numbers: candidates are compared
        as matched pairs, so the argmax is stable at small n_mc instead of being
        decided by sampling noise.
        """
        post = np.exp(logL - logL.max())
        post /= post.sum()
        hs = rng.choice(self.H, size=n_mc, p=post)

        # Candidates split by shape: 1-column block groups (vectorised together)
        # and the K-column context group (handled separately).
        single = [g for g in available if g != 0]
        scores: dict[int, float] = {}

        if single:
            cols = np.array([self.group_cols[g][0] for g in single])
            mu_s = self.MU[np.ix_(hs, cols)]                     # (S, G)
            sd_s = self.SD[np.ix_(hs, cols)]
            x = mu_s + sd_s * rng.standard_normal(mu_s.shape)
            MUc = self.MU[:, cols].T                             # (G, H)
            SDc = self.SD[:, cols].T
            LSc = self.LOGSD[:, cols].T
            d = (x[:, :, None] - MUc[None, :, :]) / SDc[None, :, :]
            ll = -0.5 * d * d - LSc[None, :, :]                  # (S, G, H)
            ent = self._entropy_y(logL[None, None, :] + ll)      # (S, G)
            for i, g in enumerate(single):
                scores[g] = -float(ent[:, i].mean())

        if 0 in available:
            cols = self.group_cols[0]
            mu_s = self.MU[np.ix_(hs, cols)]                     # (S, K)
            sd_s = self.SD[np.ix_(hs, cols)]
            x = mu_s + sd_s * rng.standard_normal(mu_s.shape)
            d = (x[:, None, :] - self.MU[None, :, cols]) / self.SD[None, :, cols]
            ll = (-0.5 * d * d - self.LOGSD[None, :, cols]).sum(axis=-1)   # (S, H)
            ent = self._entropy_y(logL[None, :] + ll)            # (S,)
            scores[0] = -float(ent.mean())

        return max(scores, key=lambda g: scores[g])

    def run_myopic(
        self,
        x: np.ndarray,
        budget: int,
        rng: np.random.Generator | None = None,
        *,
        n_mc: int | None = None,
        forced_first: int | None = None,
        free_groups: tuple[int, ...] = (),
    ) -> tuple[list[int], list[int]]:
        """Acquire `budget` groups myopically.

        Returns (groups acquired, predictions) where predictions[t] is the MAP
        label after t+1 acquisitions. A myopic policy ignores the remaining
        budget by definition, so its first t choices do not depend on the total
        -- one budget-8 run therefore yields every smaller budget as a prefix,
        exactly, with no re-run and no drift between the two rows of the table.

        `forced_first` overrides only the FIRST action, then myopic play
        resumes. That gives the non-myopic arm identical machinery to the myopic
        one, so the two differ in exactly one decision and nothing else.

        `free_groups` are observed before play at NO budget cost. Setting it to
        (0,) upper-bounds `forced_first=0`: it grants the context without
        charging a slot for it, which brackets how much of the non-myopic
        advantage is the information versus how much is the sequencing.
        """
        acquired: list[int] = []
        preds: list[int] = []
        logL = np.zeros(self.H)
        available = list(range(self.n_groups))
        for g in free_groups:
            available.remove(g)
            logL = logL + self.loglik_cols(x, self.group_cols[g])
        for t in range(budget):
            if t == 0 and forced_first is not None:
                g = forced_first
            elif n_mc is None:                      # default: exact, no sampling
                g = self.myopic_step_exact(logL, available)
            else:                                   # sampled, kept for the audit
                g = self.myopic_step(logL, available, rng, n_mc)
            available.remove(g)
            acquired.append(g)
            logL = logL + self.loglik_cols(x, self.group_cols[g])
            preds.append(int(np.argmax(self.label_posterior(logL))))
        return acquired, preds


def check_exact_vs_sampled(ds: CubeNMRepro, n_states: int = 12,
                           n_mc: int = 20000, seed: int = 0) -> dict:
    """Prove the quadrature/collapse scores match a converged sampled estimate.

    The exact scorer replaces Monte Carlo with (a) 1-D quadrature for block
    columns and (b) a 5-term collapse for the context group. Both are claims
    about an estimator, and this project has been burned by claims about
    estimators that were never executed. So: build real partially-observed
    states, score every candidate both ways, and report the worst disagreement
    plus whether the two argmaxes coincide.

    Sampling error at n_mc=20000 is itself ~1/sqrt(n), so small residual gaps
    are expected; a mis-derived quadrature would show a systematic O(0.1) gap,
    not a O(0.005) one.
    """
    bayes = CubeNMBayes(ds)
    rng = np.random.default_rng(seed)
    worst, agree, checked = 0.0, 0, 0
    for i in range(n_states):
        logL = np.zeros(bayes.H)
        available = list(range(bayes.n_groups))
        # walk into a non-trivial state so the posterior is not uniform
        for g in (1 + rng.integers(0, bayes.n_groups - 1, size=2)).tolist():
            if g in available:
                available.remove(g)
                logL = logL + bayes.loglik_cols(ds.features[i], bayes.group_cols[g])

        post = np.exp(logL - logL.max()); post /= post.sum()
        hs = rng.choice(bayes.H, size=n_mc, p=post)

        probe = [0] + available[:6]
        sampled: dict[int, float] = {}
        for g in probe:
            cols = bayes.group_cols[g]
            mu_s, sd_s = bayes.MU[np.ix_(hs, cols)], bayes.SD[np.ix_(hs, cols)]
            x = mu_s + sd_s * rng.standard_normal(mu_s.shape)
            d = (x[:, None, :] - bayes.MU[None, :, cols]) / bayes.SD[None, :, cols]
            ll = (-0.5 * d * d - bayes.LOGSD[None, :, cols]).sum(axis=-1)
            sampled[g] = -float(bayes._entropy_y(logL[None, :] + ll).mean())

        # exact scores for the same probe set
        ex: dict[int, float] = {}
        post_full = post
        sing = [g for g in probe if g != 0]
        slots = np.array([bayes._col_slot[bayes.group_cols[g][0]] for g in sing])
        joint = post_full[:, None, None] * bayes._pdf[:, slots, :]
        px = joint.sum(axis=0)
        w = px / px.sum(axis=1, keepdims=True)
        cond = joint / np.maximum(px[None, :, :], 1e-300)
        py = cond.reshape(bayes.K, N_LABELS, len(sing), -1).sum(axis=0)
        ent = -(py * np.log(np.maximum(py, 1e-300))).sum(axis=0)
        for j, g in enumerate(sing):
            ex[g] = -float((w[j] * ent[j]).sum())
        P = post_full.reshape(bayes.K, N_LABELS)
        pc = P.sum(axis=1)
        pyc = P / np.maximum(pc[:, None], 1e-300)
        ex[0] = -float((pc * -(pyc * np.log(np.maximum(pyc, 1e-300))).sum(axis=1)).sum())

        for g in probe:
            worst = max(worst, abs(ex[g] - sampled[g]))
            checked += 1
        agree += int(max(ex, key=ex.get) == max(sampled, key=sampled.get))

    return {
        "max |exact - sampled|": (worst < 0.02, round(worst, 5)),
        "argmax agrees": (agree == n_states, f"{agree}/{n_states}"),
        "candidates compared": (True, checked),
    }


def validate_likelihood(ds: CubeNMRepro, n: int = 1500) -> dict:
    """Gate: prove the hand-derived likelihood matches the actual generator.

    If MU/SD were wrong the myopic policy would be handed a broken model and
    would lose for the wrong reason.

    The load-bearing check is DOMINANCE, not an absolute accuracy threshold: a
    discriminative model fitted on fully-observed rows cannot beat the true
    posterior, so `acc_full >= acc_fitted` is a real falsification test. An
    absolute threshold is not -- the first version of this gate asserted
    acc_full > 0.97 and "failed" at 0.9693, which turned out to be the Bayes
    floor rather than a defect. CUBE-NM places the informative columns at
    {y, y+1, y+2} with y <= 7, so the windows for y=5,6,7 overlap in two of
    three columns and are irreducibly confusable; ~3% error is the construction,
    not the code. Thresholds calibrated against a guess measure the guess.

      full obs >= fitted    -> the posterior is not mis-specified
      correct block only    -> high (block identity is supplied, not inferred)
      wrong block only      -> chance, 1/8
      joint MAP on context  -> recovers the latent context from the noisy one-hot
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import train_test_split

    bayes = CubeNMBayes(ds)
    idx = np.arange(min(n, ds.n_samples))
    allc = list(range(ds.n_features))
    acc_full = np.mean([bayes.predict(ds.features[i], allc) == ds.labels[i] for i in idx])

    tr, te = train_test_split(np.arange(ds.n_samples), test_size=0.3, random_state=0)
    fitted = HistGradientBoostingClassifier(max_iter=300, random_state=0)
    fitted.fit(ds.features[tr], ds.labels[tr])
    acc_fit = float(fitted.score(ds.features[te], ds.labels[te]))
    acc_bayes_te = float(np.mean([bayes.predict(ds.features[i], allc) == ds.labels[i]
                                  for i in te]))

    def block_cols(c: int) -> list[int]:
        s = ds.n_contexts + c * BLOCK_SIZE
        return list(range(s, s + BLOCK_SIZE))

    acc_right = np.mean([
        bayes.predict(ds.features[i], block_cols(int(ds.context[i]))) == ds.labels[i]
        for i in idx])
    acc_wrong = np.mean([
        bayes.predict(ds.features[i], block_cols((int(ds.context[i]) + 1) % ds.n_contexts))
        == ds.labels[i] for i in idx])

    ctx_hits = 0
    for i in idx:
        logL = bayes.loglik_cols(ds.features[i], allc)
        p = np.exp(logL - logL.max())
        p /= p.sum()
        ctx_hits += int(np.argmax(p.reshape(bayes.K, N_LABELS).sum(axis=1)) == ds.context[i])

    return {
        "exact Bayes >= fitted model": (acc_bayes_te >= acc_fit - 0.003,
                                        f"{acc_bayes_te:.4f} vs fitted {acc_fit:.4f}"),
        "full observation near ceiling": (acc_full > 0.94, round(float(acc_full), 4)),
        "correct block high": (acc_right > 0.90, round(float(acc_right), 4)),
        "wrong block ~ chance": (abs(acc_wrong - 0.125) < 0.04, round(float(acc_wrong), 4)),
        "context MAP recovers latent": (ctx_hits / len(idx) > 0.97,
                                        round(ctx_hits / len(idx), 4)),
    }
