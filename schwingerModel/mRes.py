"""Residual mass from the domain-wall midpoint correlator.

The axial Ward identity gives, for any source and a local point sink,
    R(t) = <J5q(t) P(0)> / <P(t) P(0)>  ->  m_res   (large t)
The two correlators come from distillation.readMres (per config, by dt); this
module resamples them and, optionally, reduces the ratio curve to plateau
values. Same shape as GEVP.bootstrapEnsemble: the ratio is formed per resample
(a ratio of ensemble means, never a mean of per-config ratios), the reduce is
applied to every resample, and the result is [central, err, cov].
"""
import numpy as np
from tqdm import tqdm


def bootstrapMres(C_PP, C_JP, weights=None, reduce=None, numResamples=10000, seed=None,
                  progress=True, quantile=.68):
    """Bootstrap statistics of the ratio curve R(t) = <C_JP>/<C_PP>.

    Args:
        C_PP, C_JP: (nCfg, dimt) from distillation.readMres.
        weights: (nCfg,) reweighting factors. Defaults to None (uniform).
        reduce: Callable applied to each resample's (dimt,) ratio curve — e.g.
            plateauReduce(...) for plateau values. Defaults to None (identity,
            so the statistics are of the curve itself).
        numResamples, seed, progress, quantile: As GEVP.bootstrapEnsemble.

    Returns:
        list: [central, err, cov] — central = reduce of the weighted-mean ratio;
        err is (2, *central.shape) with rows (high - central, central - low);
        cov is the (k, k) covariance of a 1-d reduce output (None otherwise).
    """
    C_PP, C_JP = np.asarray(C_PP), np.asarray(C_JP)
    n_cfg = len(C_PP)
    if weights is None:
        weights = np.ones(n_cfg)
    if reduce is None:
        reduce = lambda R: R

    rng = np.random.default_rng(seed)
    counts = rng.multinomial(n_cfg, np.full(n_cfg, 1.0 / n_cfg), size=numResamples)
    wEff = counts * weights[None, :]                                  # (R, n_cfg)

    central = reduce((weights @ C_JP) / (weights @ C_PP))
    ratios = (wEff @ C_JP) / (wEff @ C_PP)                           # (R, dimt)
    iterator = tqdm(ratios, desc="Bootstrap reduce", leave=False) if progress else ratios
    samples = np.array([reduce(r) for r in iterator])

    low  = np.nanquantile(samples, (1-quantile)/2, axis=0)
    high = np.nanquantile(samples, (1+quantile)/2, axis=0)
    err  = np.array([high - central, central - low])

    finite = np.isfinite(samples.reshape(len(samples), -1)).all(axis=1)
    cov = np.cov(samples[finite], rowvar=False) if samples.ndim == 2 and finite.sum() >= 10 else None
    return [central, err, cov]


def plateauReduce(fitT=(5, 30)):
    """Reduce factory for bootstrapMres: plateau average of R(t) per window.

    A plain mean over each window — the bootstrap supplies the error, so no
    per-point weighting is needed, and a constant fit to a correlated curve
    would want the full covariance anyway.

    Args:
        fitT: One (lo, hi) window, or a list of them, each fitting on [lo, hi)
            in dt units. A list is the tool for the plateau check: vary lo at
            fixed hi in ONE bootstrap, and bootstrapMres's cov then holds the
            (highly correlated) window-to-window covariance.

    Returns:
        Callable: (dimt,) ratio curve -> scalar for a single window, or
        (nWindows,) for a list.
    """
    single = np.ndim(fitT[0]) == 0
    windows = [tuple(fitT)] if single else [tuple(w) for w in fitT]

    def _reduce(R):
        out = np.array([np.mean(R[lo:hi]) for lo, hi in windows])
        return out[0] if single else out

    return _reduce
