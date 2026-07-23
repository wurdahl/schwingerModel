import numpy as np
from joblib import Parallel, delayed
import joblib
from scipy.linalg import eig
from scipy.optimize import curve_fit, linear_sum_assignment
from tqdm.auto import tqdm

from . import distillation as dist
from .wick import contract, mergeFlavors
from .evaluator import evalTable


# ---------------------------------------------------------------------------
# Table-driven measurement: Interpolator basis -> per-config correlation data
# ---------------------------------------------------------------------------

def contractBasis(basis):
    """Diagram tables for every (sink, source) pair of an Interpolator basis,
    with flavors merged to their degenerate class. Computed once per analysis."""
    return {(a, b): mergeFlavors(contract(snk, src))
            for a, snk in enumerate(basis) for b, src in enumerate(basis)}


def _measureConfigWick(filePath, configIndex, tables, n):
    """Worker: evaluate all pair tables on one config's workspace.
    Returns (conn (n, n, T), disc {(a,b): (coeffs, ABcorr, A, B)}) where per
    disc diagram d: A, B are (D, T) loop series and
    ABcorr[d, dt] = (1/T) sum_t A[d, t+dt] B[d, t]."""
    ws = dist.DistillWorkspace.load(filePath, configIndex)
    T = ws.eigVecs.shape[0]
    conn = np.zeros((n, n, T), dtype=complex)
    disc = {}
    for (a, b), table in tables.items():
        res = evalTable(ws, table)
        conn[a, b] = res.conn
        if res.disc:
            coeffs = np.array([c for c, _, _ in res.disc])
            A = np.stack([Ad for _, Ad, _ in res.disc])          # (D, T)
            B = np.stack([Bd for _, _, Bd in res.disc])
            AB = np.stack([np.mean(np.roll(A, -dt, axis=-1) * B, axis=-1)
                           for dt in range(T)], axis=-1)          # (D, T)
            disc[(a, b)] = (coeffs, AB, A, B)
    return conn, disc


def measureEnsemble(filePath, configIndices, basis, n_jobs=-1):
    """
    Measure the full correlation-matrix data for an Interpolator basis over an
    ensemble cache. Symbolic contraction happens once; workers only evaluate.

    Returns {"conn": (n_cfg, n, n, T) complex,
             "disc": {(a,b): {"coeffs": (D,), "AB": (n_cfg, D, T),
                              "A": (n_cfg, D, T), "B": (n_cfg, D, T)}}}
    The disc pieces need ensemble-level vacuum subtraction — that happens in
    bootstrapEnsemble, never per config.
    """
    tables = contractBasis(basis)
    n = len(basis)
    with dist.tqdm_joblib(tqdm(total=len(configIndices), desc="Measuring configs")):
        results = Parallel(n_jobs=n_jobs)(
            delayed(_measureConfigWick)(filePath, i, tables, n) for i in configIndices)

    conn = np.array([r[0] for r in results])
    disc = {}
    for pair in results[0][1]:
        disc[pair] = {"coeffs": results[0][1][pair][0],
                      "AB": np.array([r[1][pair][1] for r in results]),
                      "A":  np.array([r[1][pair][2] for r in results]),
                      "B":  np.array([r[1][pair][3] for r in results])}
    return {"conn": conn, "disc": disc}


# ---------------------------------------------------------------------------
# Disc-aware bootstrap
# ---------------------------------------------------------------------------

def _vacSeries(Am, Bm):
    """(1/T) sum_t Am[..., t+dt] Bm[..., t] over the last axis, any leading axes."""
    T = Am.shape[-1]
    return np.stack([np.mean(np.roll(Am, -dt, axis=-1) * Bm, axis=-1)
                     for dt in range(T)], axis=-1)


def _assembleC(connMean, discMeans):
    """Combine connected means with vacuum-subtracted disc pieces into C.
    connMean: (..., n, n, T); discMeans: {(a,b): (coeffs, ABm, Am, Bm)}."""
    C = connMean.copy()
    for (a, b), (coeffs, ABm, Am, Bm) in discMeans.items():
        C[..., a, b, :] += np.einsum('d,...dt->...t', coeffs, ABm - _vacSeries(Am, Bm))
    return C


def bootstrapEnsemble(measured, weights=None, reduce=None, numResamples=10000, seed=None,
                      progress=True):
    """
    Bootstrap statistics for measureEnsemble output, with disc vacuum subtraction
    done per resample (subtraction needs ensemble means, so it lives here).

    reduce: applied to each resample's assembled (n, n, T) matrix; None = identity,
            or gevpReduce / makeGevpReduce(...) for GEVP curves.
    Returns [central, err (2, ...), cov] with the same shape conventions as
    bootstrapEnsemble2pt.
    """
    conn = measured["conn"]
    disc = measured["disc"]
    n_cfg = len(conn)
    if weights is None:
        weights = np.ones(n_cfg)
    if reduce is None:
        reduce = lambda C: C

    rng = np.random.default_rng(seed)
    counts = rng.multinomial(n_cfg, np.full(n_cfg, 1.0 / n_cfg), size=numResamples)
    wEff = counts * weights[None, :]                              # (R, n_cfg)

    def wMean(x):                                                 # central weighted mean
        return np.tensordot(weights, x, axes=(0, 0)) / weights.sum()

    def rMean(x):                                                 # all resample means at once
        flat = x.reshape(n_cfg, -1)
        m = (wEff @ flat) / wEff.sum(axis=1, keepdims=True)
        return m.reshape(numResamples, *x.shape[1:])

    discCentral = {p: (d["coeffs"], wMean(d["AB"]), wMean(d["A"]), wMean(d["B"]))
                   for p, d in disc.items()}
    discSamples = {p: (d["coeffs"], rMean(d["AB"]), rMean(d["A"]), rMean(d["B"]))
                   for p, d in disc.items()}

    centralC = _assembleC(wMean(conn), discCentral)
    samplesC = _assembleC(rMean(conn), discSamples)

    central = np.real(reduce(centralC))
    iterator = tqdm(samplesC, desc="Bootstrap reduce", leave=False) if progress else samplesC
    samples = np.real(np.array([reduce(c) for c in iterator]))

    # drop resamples where the reduce failed (e.g. massReduce fit window hit
    # non-positive values) so they don't poison percentiles and covariance
    valid = ~np.isnan(samples.reshape(len(samples), -1)).any(axis=1)
    if not valid.all():
        import warnings
        warnings.warn(f"bootstrapEnsemble: dropped {(~valid).sum()}/{len(valid)} "
                      "resamples with NaN reduce output")
        samples = samples[valid]

    low  = np.percentile(samples, 2.5,  axis=0)
    high = np.percentile(samples, 97.5, axis=0)
    err  = np.array([high - central, central - low])

    if samples.ndim == 2:
        cov = np.cov(samples, rowvar=False)
    elif samples.ndim == 3:
        cov = np.array([np.cov(samples[:, :, e], rowvar=False)
                        for e in range(samples.shape[2])])
    else:
        cov = None

    return [central, err, cov]


def gevp(corrMat, ti=1, sortBy="vector", refVecs=None):
    """
    corrMat: (n, n, dimt) symmetric correlation matrix.
    Returns newCorr (dimt-ti, n) eigenvalue curves and the reference eigenvectors.

    sortBy="value":  order eigenvalues descending at each t independently (old behavior;
                     mis-assigns states where curves approach or cross).
    sortBy="vector": track states across t by eigenvector overlap in the C(ti) metric —
                     GEVP eigenvectors are C(ti)-orthogonal, so |v_ref^dag C(ti) v(t)|
                     identifies which physical state each eigenpair belongs to.
                     State labels are fixed by descending eigenvalue at t = ti+1.
    """
    dimt = corrMat.shape[2]
    n = corrMat.shape[0]
    ref = corrMat[:, :, ti]

    gevpOutput = [eig(a=corrMat[:, :, t], b=ref) for t in range(ti, dimt)]

    if sortBy == "value":
        newCorr = np.array([np.sort(np.real(ev[0]))[::-1] for ev in gevpOutput])
        basis = np.mean([ev[1] for ev in gevpOutput], axis=0)
        return newCorr, basis

    def _refNormalize(v):
        # normalize columns in the C(ti) metric; guard vanishing norms (noise)
        nrm = np.sqrt(np.abs(np.einsum('im,ij,jm->m', v.conj(), ref, v)))
        nrm[nrm == 0] = 1.0
        return v / nrm

    if refVecs is not None:
        # external anchor (e.g. the ensemble-central eigenvectors): keeps state labels
        # consistent across bootstrap resamples instead of re-deriving them per sample
        vRef = _refNormalize(np.asarray(refVecs))
    else:
        # reference eigenvectors: at t=ti all eigenvalues are trivially 1, so label
        # states at the first nontrivial time slice, ordered by descending eigenvalue
        refIdx = 1 if len(gevpOutput) > 1 else 0
        w0, v0 = gevpOutput[refIdx]
        order0 = np.argsort(np.real(w0))[::-1]
        vRef = _refNormalize(v0[:, order0])

    newCorr = np.empty((len(gevpOutput), n))
    for k, (w, v) in enumerate(gevpOutput):
        v = _refNormalize(v)
        overlap = np.abs(vRef.conj().T @ ref @ v)          # (state, eigenpair)
        rows, cols = linear_sum_assignment(-overlap)        # maximize total overlap
        assign = cols[np.argsort(rows)]                     # eigenpair for each state
        newCorr[k] = np.real(w[assign])

    return newCorr, vRef


def gevpMassExtract(gevpStatsOut, fitT=[1,10], ti=1, eigenIdx=0, coshExpr=True):
    """
    gevpStatsOut: [mean (dimt-ti, n), errors, covMat (n, dimt-ti, dimt-ti)]
    eigenIdx: which eigenvalue to fit (0 = lowest mass, 1 = next, ...)

    Fits in log space: minimizes relative residuals, giving equal weight per decade.
    Covariance is propagated as Σ_log[i,j] = Σ_lin[i,j] / (C[i] * C[j]).
    """
    dimt = gevpStatsOut[0].shape[0] + ti

    # logA is a free amplitude: the GEVP normalization lambda(ti)=1 is a
    # convention, and pinning the fit through it pushes excited-state
    # contamination at ti into the mass. logA != 0 measures that contamination.
    def expDecay_log(nt, Energy, logA):
        return logA - nt * Energy

    def coshCorrel_log(nt, Energy, logA):
        numer = np.logaddexp(-(nt + ti) * Energy, ((nt + ti) - dimt) * Energy)
        denom = np.logaddexp(-ti * Energy, (ti - dimt) * Energy)
        return logA + numer - denom

    mean = gevpStatsOut[0][fitT[0]:fitT[1], eigenIdx]
    cov  = gevpStatsOut[2][eigenIdx, fitT[0]:fitT[1], fitT[0]:fitT[1]]

    log_mean = np.log(mean)
    inv_mean = 1.0 / mean
    log_cov  = cov * np.outer(inv_mean, inv_mean)

    model = coshCorrel_log if coshExpr else expDecay_log
    fitMass = curve_fit(model, xdata=np.arange(fitT[0], fitT[1]),
                ydata=log_mean, sigma=log_cov, absolute_sigma=True,
                p0=[0.5, 0.0], bounds=([0, -np.inf], [np.inf, np.inf]))

    # [E, dE, logA, dlogA] — dE is the profiled (marginal) mass error
    return np.array([fitMass[0][0], np.sqrt(fitMass[1][0, 0]),
                     fitMass[0][1], np.sqrt(fitMass[1][1, 1])])

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

def gevpReduce(Cmean, ti=1, refVecs=None, shift=0):
    """
    Reduce one (n, n, T) mean correlation matrix to GEVP eigenvalue curves.
    Symmetrization happens here, explicitly, on the ensemble/resample mean —
    per-config matrices are NOT hermitian, only their average is.

    shift > 0: solve the GEVP on the shifted matrix C(t+shift) - C(t), which
    annihilates t-independent thermal terms (two-particle around-the-torus
    pollution) exactly. Output has (T - shift - ti) time slices; the curves are
    no longer cosh-shaped — fit forward-exponential on early times.
    """
    Csym = 0.5 * (Cmean + np.conj(np.transpose(Cmean, (1, 0, 2))))
    if shift:
        Csym = Csym[:, :, shift:] - Csym[:, :, :-shift]
    newCorr, _ = gevp(np.real(Csym), ti=ti, refVecs=refVecs)
    return newCorr


def makeGevpReduce(ti=1, shift=0):
    """
    Stateful gevpReduce for bootstrapping: the FIRST call (which bootstrapEnsemble2pt
    makes on the full-ensemble central mean) fixes the reference eigenvectors, and every
    subsequent call (the resamples) labels its states against that anchor. This prevents
    state labels from flipping between resamples when eigenvalues are close — the cause
    of bimodal bootstrap distributions and central values outside the percentile band.
    Create a fresh instance per bootstrapEnsemble2pt call.
    """
    state = {}

    def _reduce(Cmean):
        Csym = 0.5 * (Cmean + np.conj(np.transpose(Cmean, (1, 0, 2))))
        if shift:
            Csym = Csym[:, :, shift:] - Csym[:, :, :-shift]
        if "vRef" not in state:
            curves, vRef = gevp(np.real(Csym), ti=ti)
            state["vRef"] = vRef
            return curves
        return gevp(np.real(Csym), ti=ti, refVecs=state["vRef"])[0]

    return _reduce


def _fitLogLinear(curve, fitT):
    """(energy, logA) from a log-linear fit on [fitT[0], fitT[1]).
    (nan, nan) if the window has non-positive values (signal lost to noise)."""
    y = curve[fitT[0]:fitT[1]]
    if len(y) < 2 or np.any(y <= 0) or not np.all(np.isfinite(y)):
        return np.nan, np.nan
    ts = np.arange(fitT[0], fitT[1])
    slope, intercept = np.polyfit(ts, np.log(y), 1)
    return -slope, intercept


def massReduce(ti=1, shift=0, fitT=(2, 8), withAmp=False):
    """
    Reduce for bootstrapEnsemble that goes all the way to masses: anchored GEVP
    (optionally shifted) then a two-parameter log-linear fit per state. Because
    the fit is redone on every resample, the bootstrap distribution of the mass
    exactly marginalizes the amplitude (and inherits all data correlations).
    Output per resample: (n_states,) masses -> bootstrapEnsemble returns
    [masses, err, cov] with cov the n_states x n_states mass covariance
    (useful for splittings like E_pipi - 2 E_pi).
    fitT is in curve-index units of the (shifted) GEVP output; pass one (lo, hi)
    window for all states or a list of per-state windows (excited states need
    earlier/shorter windows than the ground state).

    withAmp=False: reduce output is (n_states,) masses; cov is the mass
    covariance matrix. withAmp=True: output is (n_states, 2) with columns
    [E, logA] — the fitted curve is exp(logA - E*t) in the (shifted) curve's
    time units — and cov becomes (2, n, n): [0] mass cov, [1] logA cov.
    """
    gr = makeGevpReduce(ti=ti, shift=shift)
    perState = isinstance(fitT[0], (tuple, list))

    def _reduce(Cmean, withAmp=withAmp):
        curves = gr(Cmean)
        windows = fitT if perState else [fitT] * curves.shape[1]
        fits = np.array([_fitLogLinear(curves[:, e], w)
                         for e, w in enumerate(windows)])   # (n, 2)
        return fits if withAmp else fits[:, 0]

    return _reduce


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

