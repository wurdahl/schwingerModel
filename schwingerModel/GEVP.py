import numpy as np
from joblib import Parallel, delayed
import joblib

from . import distillation as dist
from .correlation import gevp

def build2ptCorrelationMatrix(filePath, configIndex, basis):
    ws = dist.DistillWorkspace.load(filePath, configIndex)
    n, T = len(basis), ws.modelObj.dimt
    C = np.empty((n, n, T), dtype=complex)
    for a, snk in enumerate(basis):
        for b, src in enumerate(basis):
            C[a, b] = dist.evalTwoPoint(ws, snk, src)
    return C

def measureEnsemble2pt(filePath, configIndices, basis, n_jobs=-1):
    correls = Parallel(n_jobs=n_jobs)(delayed(build2ptCorrelationMatrix)(filePath, ind, basis) for ind in configIndices)

    return np.array(correls)

def gevpReduce(Cmean, ti=1):
    """
    Reduce one (n, n, T) mean correlation matrix to GEVP eigenvalue curves (T-ti, n).
    Symmetrization happens here, explicitly, on the ensemble/resample mean —
    per-config matrices are NOT hermitian, only their average is.
    """
    Csym = 0.5 * (Cmean + np.conj(np.transpose(Cmean, (1, 0, 2))))
    newCorr, _ = gevp(np.real(Csym), ti=ti)
    return newCorr


def bootstrapEnsemble2pt(correls, weights=None, reduce=None, numResamples=10000, seed=None):
    """
    Bootstrap statistics over per-config correlation matrices.

    correls: (n_cfg, n, n, T) from measureEnsemble2pt
    weights: (n_cfg,) reweighting factors (default: uniform)
    reduce:  callable applied to each resample's weighted-mean (n, n, T) matrix.
             None -> identity (raw matrix statistics)
             lambda C: np.real(C[a, b])        -> single-correlator stats, feeds correlMassExtract
             lambda C: gevpReduce(C, ti=1)     -> GEVP curves (T-ti, n), feeds gevpMassExtract
    Returns [central, err (2, ...), cov] matching the existing fitter conventions:
             reduce output (T,)    -> cov (T, T)
             reduce output (T', n) -> cov (n, T', T')   (per eigenvalue, like GEVPStats)
             otherwise             -> cov None
    """
    correls = np.asarray(correls)
    n_cfg = len(correls)
    if weights is None:
        weights = np.ones(n_cfg)
    if reduce is None:
        reduce = lambda C: C

    # Resampling via multinomial counts: a bootstrap draw of configs (with replacement)
    # is equivalent to integer counts summing to n_cfg, so each resample's weighted mean
    # is one row of a single (numResamples, n_cfg) @ (n_cfg, n*n*T) matmul.
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(n_cfg, np.full(n_cfg, 1.0 / n_cfg), size=numResamples)
    wEff = counts * weights[None, :]                              # (R, n_cfg)

    flat = correls.reshape(n_cfg, -1)
    sampleMeans = (wEff @ flat) / wEff.sum(axis=1, keepdims=True)  # (R, n*n*T)
    sampleMeans = sampleMeans.reshape(numResamples, *correls.shape[1:])

    centralMean = np.tensordot(weights, correls, axes=(0, 0)) / weights.sum()
    central = np.real(reduce(centralMean))
    samples = np.real(np.array([reduce(m) for m in sampleMeans]))

    low  = np.percentile(samples, 2.5,  axis=0)
    high = np.percentile(samples, 97.5, axis=0)
    err  = np.array([high - central, central - low])

    if samples.ndim == 2:                                          # (R, T)
        cov = np.cov(samples, rowvar=False)
    elif samples.ndim == 3:                                        # (R, T', n): per-eigenvalue
        cov = np.array([np.cov(samples[:, :, e], rowvar=False)
                        for e in range(samples.shape[2])])
    else:
        cov = None

    return [central, err, cov]

