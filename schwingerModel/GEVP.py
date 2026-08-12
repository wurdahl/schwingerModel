from typing import NamedTuple

import numpy as np
from joblib import Parallel, delayed
import joblib
from scipy.linalg import eig, cho_factor, cho_solve
from scipy.optimize import curve_fit, linear_sum_assignment
from tqdm import tqdm

from . import distillation as dist
from .wick import contract, mergeFlavors
from .evaluator import evalTable


# ---------------------------------------------------------------------------
# Table-driven measurement: Interpolator basis -> per-config correlation data
# ---------------------------------------------------------------------------

def contractBasis(basis):
    """Diagram tables for every (sink, source) pair of an Interpolator basis.

    Flavors are merged to their degenerate class. Computed once per analysis;
    workers then only evaluate.

    Args:
        basis: Sequence of Interpolators (creation form).

    Returns:
        dict[tuple[int, int], dict]: (sinkIdx, srcIdx) -> merged diagram table
        {DiagramKey: coeff}, for all n^2 pairs.
    """
    return {(a, b): mergeFlavors(contract(snk, src))
            for a, snk in enumerate(basis) for b, src in enumerate(basis)}


def _measureConfigWick(filePath, configIndex, tables, n):
    """Worker: evaluate all pair tables on one config's workspace.

    Args:
        filePath: HDF5 distillation cache path.
        configIndex: Which config group to load.
        tables: Output of contractBasis — {(a, b): diagram table}.
        n: Basis size (number of interpolators).

    Returns:
        tuple: (conn, disc) where conn is (n, n, T) complex and disc is
        {(a, b): (coeffs (D,), ABcorr (D, T), A (D, T), B (D, T))} with, per
        disc diagram d, A/B the sink/source loop series and
        ABcorr[d, dt] = (1/T) sum_t A[d, t+dt] B[d, t].
    """
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
    """Measure correlation-matrix data for an Interpolator basis over an ensemble.

    Symbolic contraction happens once; parallel workers only evaluate. The disc
    pieces need ensemble-level vacuum subtraction — that happens in
    bootstrapEnsemble, never per config.

    Args:
        filePath: HDF5 distillation cache path (from generateDistillFile).
        configIndices: Iterable of config indices to measure.
        basis: Sequence of Interpolators (creation form) defining the n x n matrix.
        n_jobs: joblib worker count. Defaults to -1 (all cores).

    Returns:
        dict: {"conn": (n_cfg, n, n, T) complex array,
        "disc": {(a, b): {"coeffs": (D,), "AB": (n_cfg, D, T),
        "A": (n_cfg, D, T), "B": (n_cfg, D, T)}}} — pass directly to
        bootstrapEnsemble.
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
    """Translation-averaged product of two mean loop series.

    Args:
        Am: Sink loop means, shape (..., T).
        Bm: Source loop means, shape (..., T) broadcastable with Am.

    Returns:
        np.ndarray: (..., T) array with [..., dt] = (1/T) sum_t Am[..., t+dt] * Bm[..., t].
    """
    T = Am.shape[-1]
    return np.stack([np.mean(np.roll(Am, -dt, axis=-1) * Bm, axis=-1)
                     for dt in range(T)], axis=-1)


def _assembleC(connMean, discMeans):
    """Combine connected means with vacuum-subtracted disc pieces into C.

    Args:
        connMean: Connected-part means, shape (..., n, n, T).
        discMeans: {(a, b): (coeffs, ABm, Am, Bm)} mean disc pieces; the vacuum
            term _vacSeries(Am, Bm) is subtracted from ABm here.

    Returns:
        np.ndarray: Full correlation matrix C, same shape as connMean.
    """
    C = connMean.copy()
    for (a, b), (coeffs, ABm, Am, Bm) in discMeans.items():
        C[..., a, b, :] += np.einsum('d,...dt->...t', coeffs, ABm - _vacSeries(Am, Bm))
    return C


def bootstrapEnsemble(measured, weights=None, reduce=None, numResamples=10000, seed=None,
                      progress=True, quantile=.68):
    """Bootstrap statistics for measureEnsemble output.

    Disc vacuum subtraction is done per resample (subtraction needs ensemble
    means, so it lives here, never per config). A reduce that returns NaN for
    some components (e.g. a failed fit window) only degrades those components'
    statistics — the band uses nanquantile, and a warning reports
    per-component failure fractions.

    Args:
        measured: Output of measureEnsemble: {"conn": ..., "disc": ...}.
        weights: (n_cfg,) reweighting factors. Defaults to None (uniform).
        reduce: Callable applied to each resample's assembled (n, n, T) matrix —
            e.g. makeGevpReduce(...) for GEVP curves or massReduce(...) for
            masses. Defaults to None (identity).
        numResamples: Number of bootstrap resamples. Defaults to 10000.
        seed: RNG seed for reproducible resampling. Defaults to None.
        progress: Show a tqdm bar over the reduce loop. Defaults to True.
        quantile: Central coverage (0-1) of the bootstrap error band.
            Defaults to 0.68 (~1 sigma); 0.95 recovers the old 95% band.

    Returns:
        list: [central, err, cov] where central = reduce of the weighted
        ensemble mean; err is (2, *central.shape) with rows (high - central,
        central - low) from the central quantile band; cov is (T, T) for a (T,)
        reduce output, (n, T', T') per state for a (T', n) output, and None
        otherwise (or when fewer than 10 jointly-finite resamples remain).
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

    # Per-component NaN policy: a state whose fit fails (e.g. a sign-crossing
    # sinh-mode in a mixed-parity basis, or a window past the noise floor) only
    # degrades ITS OWN statistics — other states keep every resample. Covariance
    # still needs jointly-finite rows.
    finite = np.isfinite(samples)
    if not finite.all():
        import warnings
        fracBad = 1.0 - finite.reshape(len(samples), -1).mean(axis=0)
        warnings.warn("bootstrapEnsemble: NaN reduce output; per-component failure "
                      f"fractions up to {fracBad.max():.0%} "
                      f"(components failing >5%: {(fracBad > 0.05).sum()})")

    low  = np.nanquantile(samples, (1-quantile)/2,  axis=0)
    high = np.nanquantile(samples, (1+quantile)/2, axis=0)
    err  = np.array([high - central, central - low])

    validRows = finite.reshape(len(samples), -1).all(axis=1)
    covSamples = samples[validRows]
    if len(covSamples) < 10:
        cov = None
    elif samples.ndim == 2:
        cov = np.cov(covSamples, rowvar=False)
    elif samples.ndim == 3:
        cov = np.array([np.cov(covSamples[:, :, e], rowvar=False)
                        for e in range(covSamples.shape[2])])
    else:
        cov = None

    return [central, err, cov]


def gevp(corrMat, ti=1, sortBy="vector", refVecs=None, labelIdx=1):
    """Solve the generalized eigenvalue problem C(t) v = lambda(t) C(ti) v.

    Args:
        corrMat: (n, n, dimt) symmetric correlation matrix.
        ti: Reference time slice for the GEVP metric C(ti). Defaults to 1.
        sortBy: "vector" (default) tracks states across t by eigenvector overlap
            in the C(ti) metric — GEVP eigenvectors are C(ti)-orthogonal, so
            |v_ref^dag C(ti) v(t)| identifies which physical state each eigenpair
            belongs to. "value" orders eigenvalues descending at each t
            independently (old behavior; mis-assigns states where curves
            approach or cross).
        refVecs: External anchor eigenvectors (n, n), e.g. the ensemble-central
            ones — keeps state labels consistent across bootstrap resamples.
            Defaults to None (derive labels at labelIdx).
        labelIdx: Curve index (t - ti) where state labels are fixed by
            descending eigenvalue. Early anchors (1) can mislabel when a heavy
            state has a large early-time amplitude (e.g. sinh-mixed bases); a
            later anchor (3-4) orders by asymptotic energy at the cost of more
            noise in the anchor slice. Defaults to 1.

    Returns:
        tuple: (newCorr, vecs) — newCorr is (dimt - ti, n) eigenvalue curves,
        one column per tracked state; vecs is the (n, n) reference eigenvector
        matrix used for labeling (pass back as refVecs for resamples).
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
        # states at a later slice (labelIdx), ordered by descending eigenvalue
        refIdx = min(labelIdx, len(gevpOutput) - 1) if len(gevpOutput) > 1 else 0
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
    """Fit one GEVP eigenvalue curve to an exponential/cosh, in log space.

    Log-space fitting minimizes relative residuals (equal weight per decade);
    covariance is propagated as Sigma_log[i,j] = Sigma_lin[i,j] / (C[i] * C[j]).
    The amplitude logA is free: the GEVP normalization lambda(ti) = 1 is a
    convention, and logA != 0 measures excited-state contamination at ti.

    Args:
        gevpStatsOut: [mean (dimt-ti, n), errors, covMat (n, dimt-ti, dimt-ti)]
            as returned by bootstrapEnsemble with a GEVP-curve reduce.
        fitT: [lo, hi) fit window in curve-index units. Defaults to [1, 10].
        ti: GEVP reference slice used to build the curves. Defaults to 1.
        eigenIdx: Which eigenvalue to fit (0 = lowest mass, 1 = next, ...).
            Defaults to 0.
        coshExpr: Fit the periodic cosh form if True, a forward exponential if
            False (use False for shifted or parity-odd curves). Defaults to True.

    Returns:
        np.ndarray: [E, dE, logA, dlogA] — energy, its profiled (marginal)
        error, log-amplitude, and its error. All NaN (with a warning) if the
        window has no positive signal, the bootstrap cov is unavailable, or
        the fit fails — callers can plot the surviving states unconditionally.
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
    if (gevpStatsOut[2] is None or not np.all(np.isfinite(mean))
            or np.any(mean <= 0)):
        import warnings
        warnings.warn(f"gevpMassExtract: no usable signal for state {eigenIdx} "
                      f"in window {fitT}; returning NaNs")
        return np.full(4, np.nan)
    cov  = gevpStatsOut[2][eigenIdx, fitT[0]:fitT[1], fitT[0]:fitT[1]]

    log_mean = np.log(mean)
    inv_mean = 1.0 / mean
    log_cov  = cov * np.outer(inv_mean, inv_mean)

    model = coshCorrel_log if coshExpr else expDecay_log
    try:
        fitMass = curve_fit(model, xdata=np.arange(fitT[0], fitT[1]),
                    ydata=log_mean, sigma=log_cov, absolute_sigma=True,
                    p0=[0.5, 0.0], bounds=([0, -np.inf], [np.inf, np.inf]))
    except Exception as fitErr:
        import warnings
        warnings.warn(f"gevpMassExtract: fit failed for state {eigenIdx} "
                      f"in window {fitT} ({fitErr}); returning NaNs")
        return np.full(4, np.nan)

    # [E, dE, logA, dlogA] — dE is the profiled (marginal) mass error
    return np.array([fitMass[0][0], np.sqrt(fitMass[1][0, 0]),
                     fitMass[0][1], np.sqrt(fitMass[1][1, 1])])


def gevpReduce(Cmean, ti=1, refVecs=None, shift=0):
    """Reduce one (n, n, T) mean correlation matrix to GEVP eigenvalue curves.

    Symmetrization happens here, explicitly, on the ensemble/resample mean —
    per-config matrices are NOT hermitian, only their average is.

    Args:
        Cmean: (n, n, T) mean correlation matrix.
        ti: GEVP reference slice. Defaults to 1.
        refVecs: Anchor eigenvectors passed through to gevp (for consistent
            state labels across resamples). Defaults to None.
        shift: If > 0, solve the GEVP on C(t + shift) - C(t), which annihilates
            t-independent thermal terms (two-particle around-the-torus
            pollution) exactly. The curves are then no longer cosh-shaped —
            fit forward-exponential on early times. Defaults to 0.

    Returns:
        np.ndarray: (T - shift - ti, n) eigenvalue curves, one column per state.
    """
    Csym = 0.5 * (Cmean + np.conj(np.transpose(Cmean, (1, 0, 2))))
    if shift:
        Csym = Csym[:, :, shift:] - Csym[:, :, :-shift]
    newCorr, _ = gevp(np.real(Csym), ti=ti, refVecs=refVecs)
    return newCorr


def makeGevpReduce(ti=1, shift=0, labelIdx=1):
    """Stateful gevpReduce factory for bootstrapping.

    The FIRST call of the returned reduce (which bootstrapEnsemble makes on the
    full-ensemble central mean) fixes the reference eigenvectors; every
    subsequent call (the resamples) labels its states against that anchor. This
    prevents state labels from flipping between resamples when eigenvalues are
    close — the cause of bimodal bootstrap distributions and central values
    outside the percentile band. Create a fresh instance per bootstrap call.

    Args:
        ti: GEVP reference slice. Defaults to 1.
        shift: Shift for C(t + shift) - C(t); see gevpReduce. Defaults to 0.
        labelIdx: Anchor curve index for state labeling; see gevp. Defaults to 1.

    Returns:
        Callable[[np.ndarray], np.ndarray]: Reduce mapping a (n, n, T) mean
        matrix to (T - shift - ti, n) anchored eigenvalue curves.
    """
    state = {}

    def _reduce(Cmean):
        Csym = 0.5 * (Cmean + np.conj(np.transpose(Cmean, (1, 0, 2))))
        if shift:
            Csym = Csym[:, :, shift:] - Csym[:, :, :-shift]
        if "vRef" not in state:
            curves, vRef = gevp(np.real(Csym), ti=ti, labelIdx=labelIdx)
            state["vRef"] = vRef
            return curves
        return gevp(np.real(Csym), ti=ti, refVecs=state["vRef"])[0]

    return _reduce


def shrinkCov(cov, shrink):
    """(1-s)*Sigma + s*diag(Sigma): pull a sample covariance toward its diagonal.

    A bootstrap covariance estimated from n_cfg configs over a k-point window is
    only well conditioned when n_cfg >> k. Below roughly n_cfg/k ~ 10 the
    off-diagonal structure is largely noise, and a correlated fit that trusts it
    overfits: it reports smaller errors than its own scatter across independent
    subsets, and picks up a small bias. Shrinking toward the diagonal keeps most
    of the gain without that failure mode. s = 0 is pure GLS, s = 1 is
    variance-only weighting.
    """
    if not shrink:
        return cov
    return (1.0 - shrink) * cov + shrink * np.diag(np.diag(cov))


def _glsSolve(X, y, cov):
    """Generalised least squares beta = (X^T S^-1 X)^-1 X^T S^-1 y.

    Falls back to variance-only weights if the Cholesky fails, so a singular
    window degrades to the diagonal fit rather than returning NaN.
    """
    try:
        c = cho_factor(cov)
        invX, invY = cho_solve(c, X), cho_solve(c, y)
    except Exception:
        d = np.diag(cov).astype(float).copy()
        d[~(d > 0)] = np.inf
        invX, invY = X / d[:, None], y / d
    return np.linalg.solve(X.T @ invX, X.T @ invY)


def _fitLogLinear(curve, fitT, logCov=None):
    """Two-parameter log-linear fit of one eigenvalue curve.

    Args:
        curve: (T',) eigenvalue curve.
        fitT: (lo, hi) window in curve-index units; fits on [lo, hi).
        logCov: (k, k) covariance of log(curve) over the window, or None for an
            unweighted fit. The model is linear in log space, so the weighted
            case is a closed-form GLS solve rather than an optimisation.

    Returns:
        tuple[float, float]: (energy, logA) so that curve ~ exp(logA - energy * t);
        (nan, nan) if the window has non-positive or non-finite values (signal
        lost to noise) or fewer than 2 points.
    """
    y = curve[fitT[0]:fitT[1]]
    if len(y) < 2 or np.any(y <= 0) or not np.all(np.isfinite(y)):
        return np.nan, np.nan
    ts = np.arange(fitT[0], fitT[1])
    if logCov is None:
        slope, intercept = np.polyfit(ts, np.log(y), 1)
        return -slope, intercept
    #columns [1, -t] so beta is (logA, energy) directly
    X = np.column_stack([np.ones_like(ts, dtype=float), -ts.astype(float)])
    logA, energy = _glsSolve(X, np.log(y), logCov)
    return energy, logA


def _backwardFactor(E, ts, ti, dimt, shift, sign):
    """The periodic image's contribution relative to the forward exponential.

    A single state on a periodic lattice contributes both e^{-E tau} and its
    around-the-torus image, so writing the curve as (forward) x (this factor)
    keeps E and logA meaning the same thing for every fit form:

        exp:  1
        cosh: 1 + e^{-E(T - 2 tau)}          symmetric, sign = +1
        sinh: 1 - e^{-E(T - 2 tau - shift)}  antisymmetric, sign = -1

    The sinh case is what a shift produces: C(tau+s) - C(tau) of a cosh is a
    sinh centred at (T - s)/2, so the factor passes through zero there and the
    curve changes sign.

    Args:
        E: Energy.
        ts: (k,) curve-index times.
        ti: GEVP reference slice, so actual time is tau = t + ti.
        dimt: Full temporal extent T.
        shift: The shift used to build the curve.
        sign: +1 for cosh, -1 for sinh, 0 for a pure forward exponential.

    Returns:
        np.ndarray: The factor at each time, with |.| applied so the log-space
        fit survives past a sinh's zero crossing.
    """
    if sign == 0:
        return np.ones_like(np.asarray(ts, dtype=float))
    tau = np.asarray(ts, dtype=float) + ti
    return np.abs(1.0 + sign * np.exp(-E * (dimt - 2.0 * tau - shift)))


_FIT_SIGNS = {"exp": 0, "cosh": 1, "sinh": -1}


def _fitPeriodic(curve, fitT, ti, dimt, shift, sign, logCov=None):
    """Two-parameter fit of one eigenvalue curve including its periodic image.

    Same (E, logA) parameterisation as _fitLogLinear — the model is
    exp(logA - E t) * _backwardFactor(...) — so the extra term costs no extra
    parameter. Fitted in log space, like the exponential case, and seeded from
    the log-linear fit.

    Args:
        curve: (T',) eigenvalue curve.
        fitT: (lo, hi) window in curve-index units; fits on [lo, hi).
        ti: GEVP reference slice.
        dimt: Full temporal extent T.
        shift: The shift used to build the curve.
        sign: +1 cosh, -1 sinh.

    Returns:
        tuple[float, float]: (energy, logA); (nan, nan) if the window has no
        usable positive signal or the fit fails.
    """
    y = curve[fitT[0]:fitT[1]]
    if len(y) < 2 or np.any(y <= 0) or not np.all(np.isfinite(y)):
        return np.nan, np.nan
    E0, logA0 = _fitLogLinear(curve, fitT, logCov)   # exponential seed
    if not np.isfinite(E0):
        return np.nan, np.nan
    ts = np.arange(fitT[0], fitT[1])

    def logModel(t, E, logA):
        with np.errstate(divide="ignore"):
            return logA - E * t + np.log(_backwardFactor(E, t, ti, dimt, shift, sign))

    # E >= 0: the periodic forms are exactly degenerate under E -> -E (the
    # curve is symmetric/antisymmetric about its centre), so a window reaching
    # past the centre can otherwise converge to the mirror solution. Same bound
    # gevpMassExtract uses. The seed can be negative there (log-linear slope of
    # a rising curve), so clip it into the feasible region.
    #a 2-D sigma is a covariance matrix to curve_fit, which whitens with it
    try:
        p, _ = curve_fit(logModel, ts, np.log(y), p0=[max(E0, 1e-6), logA0],
                         bounds=([0, -np.inf], [np.inf, np.inf]), maxfev=20000,
                         sigma=logCov, absolute_sigma=logCov is not None)
    except Exception:
        if logCov is None:
            return np.nan, np.nan
        #a singular window should degrade to the unweighted fit, not to NaN
        try:
            p, _ = curve_fit(logModel, ts, np.log(y), p0=[max(E0, 1e-6), logA0],
                             bounds=([0, -np.inf], [np.inf, np.inf]), maxfev=20000)
        except Exception:
            return np.nan, np.nan
    return p[0], p[1]


def massReduce(ti=1, shift=0, fitT=(2, 8), withAmp=False, labelIdx=1, fitForm="exp",
               cov=None, covShrink=0.0):
    """Reduce factory for bootstrapEnsemble that goes all the way to masses.

    Anchored GEVP (optionally shifted), then a two-parameter fit per state.
    Because the fit is redone on every resample, the bootstrap distribution of
    the mass exactly marginalizes the amplitude (and inherits all data
    correlations). Via bootstrapEnsemble the mass covariance is useful for
    splittings like E_pipi - 2 E_pi.

    Args:
        ti: GEVP reference slice. Defaults to 1.
        shift: Shift for C(t + shift) - C(t); see gevpReduce. Defaults to 0.
        fitT: Fit window(s) in curve-index units of the (shifted) GEVP output:
            one (lo, hi) pair for all states, or a list of per-state pairs
            (excited states need earlier/shorter windows than the ground
            state). Defaults to (2, 8).
        withAmp: If False, the reduce returns (n_states,) masses and
            bootstrapEnsemble's cov is the (n, n) mass covariance. If True, it
            returns (n_states, 2) with columns [E, logA] — the fitted curve is
            exp(logA - E * t) times the periodic factor for this fitForm, in the
            (shifted) curve's time units — and cov becomes (2, n, n): [0] mass
            cov, [1] logA cov. Defaults to False.
        labelIdx: Anchor curve index for state labeling; see gevp. Defaults to 1.
        fitForm: Which single-state shape to fit, all two-parameter:
            "exp"  forward exponential only. Correct just where the periodic
                   image is negligible, so it needs an early, short window.
            "cosh" symmetric periodic form, for shift = 0 with operators whose
                   correlator is symmetric about T/2.
            "sinh" antisymmetric form centred at (T - shift)/2, which is what a
                   shift > 0 produces from a cosh.
            "auto" "sinh" when shift > 0, else "cosh".
            The periodic forms cost no extra parameter and let the window run to
            much larger t, which matters most on short lattices. Defaults to
            "exp", preserving the previous behaviour.
        cov: (n_states, T', T') covariance of the GEVP curves, i.e. element [2]
            of a prior bootstrapEnsemble run with makeGevpReduce at the SAME
            ti/shift. Given, the plateau fits become correlated (GLS) instead of
            unweighted, which on this repo's ensembles narrows the mass error by
            1.3-2.2x at no extra cost -- the same data, better weighted. The
            weight matrix is built once from the central curve, so every
            resample is fit with identical weights. Defaults to None
            (unweighted, the previous behaviour).
        covShrink: Pull cov toward its diagonal by this fraction before use; see
            shrinkCov. Use ~0.2-0.3 when n_cfg over the window length falls
            below ~10 (e.g. the 100-config Nx=48/64 caches on a 23-point
            window), where an unshrunk correlated fit reports errors smaller
            than its own subset-to-subset scatter. Defaults to 0.0.

    Returns:
        Callable[[np.ndarray], np.ndarray]: Reduce mapping a (n, n, T) mean
        matrix to (n_states,) masses (or (n_states, 2) with withAmp);
        failed fits yield NaN for that state only.

    Raises:
        ValueError: If fitForm is not one of the names above.
    """
    if fitForm == "auto":
        fitForm = "sinh" if shift else "cosh"
    if fitForm not in _FIT_SIGNS:
        raise ValueError(f"fitForm {fitForm!r} not in {sorted(_FIT_SIGNS)} or 'auto'")
    sign = _FIT_SIGNS[fitForm]

    gr = makeGevpReduce(ti=ti, shift=shift, labelIdx=labelIdx)
    perState = isinstance(fitT[0], (tuple, list))
    state = {}

    def _logCovs(curves, windows):
        """Per-state log-space covariance over each fit window, built once.

        Sigma_log[i,j] = Sigma_lin[i,j] / (lam_i * lam_j) to first order, taken
        on the CENTRAL curve so the weight matrix is the same for every
        resample -- a per-resample weight matrix would make the estimator
        depend on its own noise.
        """
        out = []
        for e, w in enumerate(windows):
            lam = curves[w[0]:w[1], e]
            sub = np.asarray(cov)[e][w[0]:w[1], w[0]:w[1]]
            if np.any(lam <= 0) or not np.all(np.isfinite(sub)):
                out.append(None)                     # no signal: fit unweighted
                continue
            out.append(shrinkCov(sub / np.outer(lam, lam), covShrink))
        return out

    def _reduce(Cmean, withAmp=withAmp):
        dimt = Cmean.shape[2]                       # full extent, before shifting
        curves = gr(Cmean)
        windows = fitT if perState else [fitT] * curves.shape[1]
        #bootstrapEnsemble calls the reduce on the central mean first, so this
        #is where the covariance is anchored -- the same hook makeGevpReduce
        #uses for its state-label reference vectors
        if "logCov" not in state:
            state["logCov"] = (_logCovs(curves, windows) if cov is not None
                               else [None] * curves.shape[1])
        lc = state["logCov"]
        if sign == 0:
            fits = np.array([_fitLogLinear(curves[:, e], w, lc[e])
                             for e, w in enumerate(windows)])   # (n, 2)
        else:
            fits = np.array([_fitPeriodic(curves[:, e], w, ti, dimt, shift, sign, lc[e])
                             for e, w in enumerate(windows)])   # (n, 2)
        return fits if withAmp else fits[:, 0]

    return _reduce


class MassFit(NamedTuple):
    """Output of bootstrapMasses: both bootstrap passes, plus what it chose.

    masses and curves each carry the [central, err, cov] that bootstrapEnsemble
    returns, so they drop straight into the plotting helpers -- `curves` is what
    a bootstrapEnsemble(reduce=makeGevpReduce(...)) call used to produce and
    `masses` what a massReduce call produced.
    """
    masses: list        # [central, err, cov] of the plateau fits
    curves: list        # [central, err, cov] of the GEVP principal correlators
    covShrink: float    # shrinkage actually applied (see shrinkCov)
    correlated: bool    # whether the plateau fits ended up weighted at all


def bootstrapMasses(measured, ti=1, shift=0, fitT=(2, 8), fitForm="exp", withAmp=False,
                    labelIdx=1, weights=None, numResamples=10000, covResamples=None,
                    seed=None, progress=True, quantile=0.68,
                    correlated=True, covShrink="auto"):
    """GEVP curves and correlated plateau masses in one call.

    Runs the two bootstrap passes that a correlated fit needs and hides the
    plumbing between them: the first pass reduces to principal correlators and
    yields their covariance, the second refits the plateaus using that
    covariance as fixed weights.

    Why two passes, and why the weights are fixed: a single resample is one
    curve, so it carries no covariance of its own -- there is nothing to
    estimate from inside the loop. The weight matrix is a *choice of estimator*,
    like the fit window or ti, fixed once from the full ensemble and applied
    identically to every resample. Letting it vary per resample would make the
    estimator depend on its own noise.

    This does not fix anything. An unweighted fit bootstrapped this way already
    has honest errors; a correlated one is simply a lower-variance estimator of
    the same quantity, worth roughly 1.3-2x in mass error on this repo's
    ensembles -- the same data, better weighted. Pass correlated=False to get
    exactly the old behaviour.

    Args:
        measured: Output of measureEnsemble.
        ti, shift, labelIdx: GEVP settings; see gevpReduce and gevp.
        fitT, fitForm, withAmp: Plateau fit settings; see massReduce.
        weights: (n_cfg,) reweighting factors, applied to BOTH passes so the
            covariance describes the same (reweighted) ensemble as the fit.
        numResamples: Resamples for the mass pass. Defaults to 10000.
        covResamples: Resamples for the covariance pass. Defaults to
            min(numResamples, 2000) -- the covariance converges long before the
            quantiles do, so there is no reason to pay full price twice.
        seed, progress, quantile: As bootstrapEnsemble.
        correlated: False skips the covariance pass entirely and reproduces an
            unweighted massReduce exactly. Defaults to True.
        covShrink: Shrinkage toward the diagonal; see shrinkCov. "auto" uses
            windowLength / n_cfg, which is the scale at which a sample
            covariance stops being trustworthy: ~0.05 on the 400-config caches
            (essentially pure GLS) and ~0.2 on the 100-config Nx=48/64 ones,
            where an unshrunk correlated fit reports errors below its own
            subset-to-subset scatter. Defaults to "auto".

    Returns:
        MassFit: .masses, .curves, .covShrink, .correlated.
    """
    nCfg = len(measured["conn"])

    curves = bootstrapEnsemble(
        measured, weights=weights,
        reduce=makeGevpReduce(ti=ti, shift=shift, labelIdx=labelIdx),
        numResamples=covResamples or min(numResamples, 2000),
        seed=seed, progress=progress, quantile=quantile)

    #cov is None when too few resamples stayed jointly finite; that is a signal
    #the curves are too noisy to weight with, so fall back to the plain fit
    cov = curves[2] if correlated else None
    if covShrink == "auto":
        windows = fitT if isinstance(fitT[0], (tuple, list)) else [fitT]
        width = max(hi - lo for lo, hi in windows)
        covShrink = 0.0 if cov is None else min(0.9, width / max(nCfg, 1))

    masses = bootstrapEnsemble(
        measured, weights=weights,
        reduce=massReduce(ti=ti, shift=shift, fitT=fitT, withAmp=withAmp,
                          labelIdx=labelIdx, fitForm=fitForm,
                          cov=cov, covShrink=covShrink),
        numResamples=numResamples, seed=seed, progress=progress, quantile=quantile)

    return MassFit(masses=masses, curves=curves,
                   covShrink=float(covShrink), correlated=cov is not None)



