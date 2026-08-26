from typing import NamedTuple

import numpy as np
from joblib import Parallel, delayed
import joblib
from scipy.linalg import eig
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


def gevp(corrMat, ti=1, sortBy="vector", refVecs=None, labelIdx=None):
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
        labelIdx: Absolute time slice where state labels are fixed by
            descending eigenvalue. Early anchors (ti + 1) can mislabel when a
            heavy state has a large early-time amplitude (e.g. sinh-mixed
            bases); a later anchor (ti + 3 or 4) orders by asymptotic energy at
            the cost of more noise in the anchor slice. Defaults to None,
            meaning ti + 1.

    Returns:
        tuple: (newCorr, vecs) — newCorr is (dimt, n) eigenvalue curves indexed
        by absolute time t (so newCorr[ti] == 1 for every state, and t < ti is
        the growing side where excited states dominate), one column per tracked
        state; vecs is the (n, n) reference eigenvector matrix used for labeling
        (pass back as refVecs for resamples).
    """
    dimt = corrMat.shape[2]
    n = corrMat.shape[0]
    ref = corrMat[:, :, ti]

    gevpOutput = [eig(a=corrMat[:, :, t], b=ref) for t in range(dimt)]

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
        # states at a later slice (labelIdx, absolute), ordered by descending eigenvalue
        refIdx = min(ti + 1 if labelIdx is None else labelIdx, dimt - 1)
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
        gevpStatsOut: [mean (dimt, n), errors, covMat (n, dimt, dimt)]
            as returned by bootstrapEnsemble with a GEVP-curve reduce, indexed
            by absolute time.
        fitT: [lo, hi) fit window in absolute time. Defaults to [1, 10].
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
    dimt = gevpStatsOut[0].shape[0]
    lo, hi = fitT

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
        fitMass = curve_fit(model, xdata=np.arange(lo, hi) - ti,   # model variable is t - ti
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
        np.ndarray: (T - shift, n) eigenvalue curves indexed by absolute time,
        one column per state.
    """
    Csym = 0.5 * (Cmean + np.conj(np.transpose(Cmean, (1, 0, 2))))
    if shift:
        Csym = Csym[:, :, shift:] - Csym[:, :, :-shift]
    newCorr, _ = gevp(np.real(Csym), ti=ti, refVecs=refVecs)
    return newCorr


def makeGevpReduce(ti=1, shift=0, labelIdx=None):
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
        labelIdx: Absolute anchor slice for state labeling; see gevp. Defaults to
            None (ti + 1).

    Returns:
        Callable[[np.ndarray], np.ndarray]: Reduce mapping a (n, n, T) mean
        matrix to (T - shift, n) anchored eigenvalue curves indexed by absolute
        time (row ti is identically 1).
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


def svdCut(cov, cut):
    """Floor the small eigenvalues of a covariance's correlation matrix.

    A bootstrap covariance estimated from n_cfg configs over a k-point window
    is only trustworthy when n_cfg >> k. Its large eigenvalues come out about
    right, but the small ones are biased low (Wishart noise), and Sigma^-1 then
    puts enormous weight on exactly those directions -- combinations of points
    the data claim to know far better than they do. A correlated fit that
    trusts them overfits: errors below its own subset-to-subset scatter, and
    unstable refits across resamples.

    This is the regulator lsqfit/corrfitter use: diagonalise the CORRELATION
    matrix (so the scale of each point is removed), raise every eigenvalue below
    cut * lambda_max up to that floor, and transform back. Only the unreliable
    directions are touched and only ever toward LARGER uncertainty, so it is
    conservative by construction; the well-measured neighbouring-t correlations
    are left intact, which is what plain shrinkage toward the diagonal throws
    away.

    Args:
        cov: (k, k) covariance.
        cut: Floor as a fraction of the largest correlation eigenvalue. 0 or
            None returns cov unchanged (pure GLS); ~1e-3 to 1e-2 is the usual
            range, larger when n_cfg / k is small.

    Returns:
        np.ndarray: The regulated (k, k) covariance.
    """
    if not cut:
        return cov
    d = np.sqrt(np.diag(cov))
    d[~(d > 0)] = 1.0                   # a zero-variance row is left as is
    corr = cov / np.outer(d, d)
    w, v = np.linalg.eigh(corr)
    w = np.maximum(w, cut * w.max())
    return (v * w) @ v.T * np.outer(d, d)


_svdCut = svdCut        # massReduce/bootstrapMasses take a kwarg of the same name


def _fitLogLinear(curve, fitT, ti=1):
    """Unweighted log-linear fit of one eigenvalue curve, used as a fit seed.

    Args:
        curve: (T,) eigenvalue curve indexed by absolute time.
        fitT: (lo, hi) window in absolute time; fits on [lo, hi).
        ti: GEVP reference slice. The fit variable is t - ti, so logA is the
            log-amplitude at ti (0 for an uncontaminated state). Defaults to 1.

    Returns:
        tuple[float, float]: (energy, logA) so that
        curve ~ exp(logA - energy * (t - ti)), from the positive finite points
        of the window only; (nan, nan) if fewer than 2 remain.
    """
    ts = np.arange(fitT[0], fitT[1])
    y = curve[fitT[0]:fitT[1]]
    ok = np.isfinite(y) & (y > 0)
    if ok.sum() < 2:
        return np.nan, np.nan
    slope, intercept = np.polyfit(ts[ok] - ti, np.log(y[ok]), 1)
    return -slope, intercept


_FIT_SIGNS = {"exp": 0, "cosh": 1, "sinh": -1}


def _twoExpModel(ti, dimt, shift, sign):
    """The three-parameter shape massReduce fits to a GEVP principal correlator.

    A principal correlator normalised at ti carries a ground state plus, at
    early times, whatever excited-state admixture the GEVP could not resolve.
    Two states, each with its periodic image, written so the normalisation
    lambda(ti) = 1 is built in rather than fitted:

        g_E(t)    = e^{-E (t - ti)} (1 + s e^{-E (T - 2t - shift)})
        lambda(t) = [(1 - A) g_m(t) + A g_{m+dm}(t)] / [(1 - A) g_m(ti) + A g_{m+dm}(ti)]

    with s = 0 / +1 / -1 for exp / cosh / sinh. Parameters [m, dm, A]: the
    ground-state mass, the gap to the contaminating state, and its weight at
    ti (fitted within [-1, 1/2]: the state at ti is mostly the ground state).
    For t < ti the excited term GROWS fastest, which is what pins dm.

    Args:
        ti: GEVP reference slice.
        dimt: Full temporal extent T of the (unshifted) lattice.
        shift: The shift used to build the curve.
        sign: 0 forward exponential, +1 cosh, -1 sinh.

    Returns:
        Callable[[array, float, float, float], array]: f(t, m, dm, A) over
        absolute time t, with a `.periodic` flag for the plots and a
        `.ground(t, m, dm, A)` attribute giving the ground-state term alone,
        so that f / f.ground = 1 + A/(1-A) g_{m+dm}/g_m is the contamination.
    """
    def g(E, t):
        f = np.exp(-E * (t - ti))
        if sign:
            f = f * (1.0 + sign * np.exp(-E * (dimt - 2.0 * t - shift)))
        return f

    def model(t, m, dm, A):
        t = np.asarray(t, dtype=float)
        num = (1.0 - A) * g(m, t) + A * g(m + dm, t)
        den = (1.0 - A) * g(m, ti) + A * g(m + dm, ti)
        return num / den

    def ground(t, m, dm, A):
        """The ground-state term alone, with the fit's normalisation."""
        t = np.asarray(t, dtype=float)
        den = (1.0 - A) * g(m, ti) + A * g(m + dm, ti)
        return (1.0 - A) * g(m, t) / den

    model.periodic = sign != 0
    model.ground = ground
    return model


def _fitTwoExp(curve, fitT, ti, dimt, shift, sign, preTi=1, cov=None, p0=None):
    """Direct (linear-space) fit of one eigenvalue curve to _twoExpModel.

    Fitted on the data itself, not its log: a noisy point that has fluctuated
    negative is a point with a large residual, not a NaN that kills the state.
    The fitted points are the slices preTi <= t < ti plus the window [lo, hi);
    t = ti is always dropped, since lambda(ti) = 1 exactly on every resample
    and carries no information (and a zero-variance row in cov).

    Args:
        curve: (T,) eigenvalue curve indexed by absolute time.
        fitT: (lo, hi) window in absolute time, lo > ti.
        ti, dimt, shift, sign: As _twoExpModel.
        preTi: First pre-ti slice to include, or None for none. Defaults to 1:
            t = 0 is a contact term on the lattice (and, shifted, C(1) - C(0)
            has the wrong sign), not part of the state tower.
        cov: (k, k) covariance over the fitted points, in their order, already
            regulated; or None. Without it the fit is weighted by |y| — a
            relative-error weighting, first-order equivalent to the old
            log-space fit — because a truly unweighted linear fit would be
            dominated by the largest points and ignore the plateau.
        p0: Starting [m, dm, A], e.g. the central fit when refitting a
            resample; None seeds from a log-linear fit of the window. A
            resample sits a few percent from the central solution, so starting
            there removes the slow tail of resamples that otherwise wander
            along the A bound for hundreds of iterations.

    Returns:
        np.ndarray: [m, dm, A]; all NaN if fewer than 4 finite points remain or
        the optimiser fails.
    """
    ts = _fitTimes(fitT, ti, preTi)
    y = curve[ts]
    ok = np.isfinite(y)
    if ok.sum() < 4:
        return np.full(3, np.nan)
    if cov is not None and not ok.all():
        return np.full(3, np.nan)          # weights were built for the full set
    ts, y = ts[ok], y[ok]

    lower, upper = [0, 0, -1], [np.inf, np.inf, 0.5]
    if p0 is None or not np.all(np.isfinite(p0)):
        m0, _ = _fitLogLinear(curve, fitT, ti)
        if not np.isfinite(m0) or m0 <= 0:
            m0 = 0.5
        p0 = [m0, m0, 0.1]
    # a central fit that ended on a bound seeds just inside it
    p0 = np.clip(p0, np.array(lower) + 1e-9, np.array(upper) - 1e-9)

    if cov is None:
        sigma = np.maximum(np.abs(y), 1e-3 * np.max(np.abs(y)))   # floor near a sinh crossing
    else:
        sigma = cov
    model = _twoExpModel(ti, dimt, shift, sign)
    # Bounds are what make the parameters identifiable: (A, dm) -> (1 - A, -dm)
    # is the same curve, so dm >= 0 fixes which state is "excited"; and at
    # A = 1 the ground term vanishes and m is free, so A <= 1/2 keeps the state
    # at ti mostly the ground state. m >= 0 removes the periodic forms' E -> -E
    # mirror.
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            # 1e-6 tolerances: the default 1e-8 costs ~10x the evaluations on
            # resamples where A ~ 0 leaves dm unconstrained (a flat valley the
            # optimiser creeps along), for parameter changes far below any
            # bootstrap error. maxfev is a safety net, not a convergence knob.
            p, _ = curve_fit(model, ts, y, p0=p0, sigma=sigma, absolute_sigma=True,
                             bounds=(lower, upper), maxfev=2000, ftol=1e-6, xtol=1e-6)
    except Exception:
        return np.full(3, np.nan)
    return np.asarray(p, dtype=float)


def _fitTimes(fitT, ti, preTi):
    """Absolute times a fit uses: slices preTi <= t < ti (if preTi is not None)
    plus the window [lo, hi), never t = ti."""
    ts = np.arange(fitT[0], fitT[1])
    if preTi is not None:
        ts = np.concatenate([np.arange(preTi, ti), ts])
    return ts[ts != ti]


def massReduce(ti=1, shift=0, fitT=(2, 8), withAmp=False, labelIdx=None, fitForm="exp",
               preTi=1, cov=None, svdCut=None):
    """Reduce factory for bootstrapEnsemble that goes all the way to masses.

    Anchored GEVP (optionally shifted), then a three-parameter two-exponential
    fit per state (see _twoExpModel), done directly on the curve rather than
    its log. Because the fit is redone on every resample, the bootstrap
    distribution of the mass exactly marginalizes the other parameters (and
    inherits all data correlations). Via bootstrapEnsemble the mass covariance
    is useful for splittings like E_pipi - 2 E_pi.

    Args:
        ti: GEVP reference slice. Defaults to 1.
        shift: Shift for C(t + shift) - C(t); see gevpReduce. Defaults to 0.
        fitT: Fit window(s) in absolute time on the (shifted) GEVP output,
            lo > ti: one (lo, hi) pair for all states, or a list of per-state
            pairs (excited states need earlier/shorter windows than the ground
            state). The pre-ti slices are added automatically; see preTi.
            Defaults to (2, 8).
        withAmp: If False, the reduce returns (n_states,) masses and
            bootstrapEnsemble's cov is the (n, n) mass covariance. If True, it
            returns (n_states, 3) with columns [m, dm, A] (see _twoExpModel)
            and cov becomes (3, n, n). Defaults to False.
        labelIdx: Absolute anchor slice for state labeling; see gevp. Defaults to
            None (ti + 1).
        fitForm: Which single-state shape each of the two states has:
            "exp"  forward exponential only. Correct just where the periodic
                   image is negligible, so it needs an early, short window.
            "cosh" symmetric periodic form, for shift = 0 with operators whose
                   correlator is symmetric about T/2.
            "sinh" antisymmetric form centred at (T - shift)/2, which is what a
                   shift > 0 produces from a cosh. The fit is direct, so the
                   window may run through the sign change.
            "auto" "sinh" when shift > 0, else "cosh".
            Defaults to "exp".
        preTi: First of the t < ti slices to fit as well -- where the excited
            state dominates and dm is actually constrained -- or None to fit
            the window only (then the third parameter is essentially
            unconstrained and the fit degenerates; keep it on). Defaults to 1:
            t = 0 is a contact term and must stay out.
        cov: (n_states, T, T) covariance of the GEVP curves, i.e. element [2]
            of a prior bootstrapEnsemble run with makeGevpReduce at the SAME
            ti/shift. Given, the fits become correlated (GLS) instead of
            |y|-weighted. The weight matrix is built once from the central
            curve, so every resample is fit with identical weights. Defaults
            to None.
        svdCut: Regulator for each state's fit-point covariance; see the svdCut
            function. Defaults to None (pure GLS).

    Returns:
        Callable[[np.ndarray], np.ndarray]: Reduce mapping a (n, n, T) mean
        matrix to (n_states,) masses (or (n_states, 3) with withAmp);
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

    def _covs(windows):
        """Per-state covariance over each state's fit points, built once."""
        out = []
        for e, w in enumerate(windows):
            ts = _fitTimes(w, ti, preTi)
            sub = np.asarray(cov)[e][np.ix_(ts, ts)]
            if not np.all(np.isfinite(sub)):
                out.append(None)                     # no signal: fit |y|-weighted
                continue
            out.append(_svdCut(sub, svdCut))
        return out

    def _reduce(Cmean, withAmp=withAmp):
        dimt = Cmean.shape[2]                       # full extent, before shifting
        curves = gr(Cmean)
        windows = fitT if perState else [fitT] * curves.shape[1]
        #bootstrapEnsemble calls the reduce on the central mean first, so this
        #is where the covariance is anchored -- the same hook makeGevpReduce
        #uses for its state-label reference vectors -- and where the central
        #fit is kept to seed every resample's fit
        first = "cov" not in state
        if first:
            state["cov"] = (_covs(windows) if cov is not None
                            else [None] * curves.shape[1])
        cs = state["cov"]
        seeds = [None] * curves.shape[1] if first else state["central"]
        fits = np.array([_fitTwoExp(curves[:, e], w, ti, dimt, shift, sign, preTi, cs[e], seeds[e])
                         for e, w in enumerate(windows)])   # (n, 3)
        if first:
            state["central"] = list(fits)
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
    svdCut: float       # SVD cut actually applied (see svdCut)
    correlated: bool    # whether the plateau fits ended up weighted at all


def bootstrapMasses(measured, ti=1, shift=0, fitT=(2, 8), fitForm="exp", withAmp=False,
                    preTi=1, labelIdx=None, weights=None, numResamples=2000, covResamples=None,
                    seed=None, progress=True, quantile=0.68,
                    correlated=True, svdCut=1e-3):
    """GEVP curves and correlated plateau masses in one call.

    Runs the two bootstrap passes that a correlated fit needs and hides the
    plumbing between them: the first pass reduces to principal correlators and
    yields their covariance, the second refits the curves using that
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
        fitT, fitForm, withAmp, preTi: Fit settings; see massReduce.
        weights: (n_cfg,) reweighting factors, applied to BOTH passes so the
            covariance describes the same (reweighted) ensemble as the fit.
        numResamples: Resamples for the mass pass. Defaults to 2000: a 68%
            quantile from 2000 resamples carries a ~3% relative error on the
            error bar, below anything read off a plot, at a fifth of the cost
            of bootstrapEnsemble's 10000 (every resample is a GEVP plus a
            fit per state).
        covResamples: Resamples for the covariance pass. Defaults to
            min(numResamples, 2000) -- the covariance converges long before the
            quantiles do, so there is no reason to pay full price twice.
        seed, progress, quantile: As bootstrapEnsemble.
        correlated: False skips the covariance pass entirely and reproduces an
            unweighted massReduce exactly. Defaults to True.
        svdCut: Floor for the small eigenvalues of each window's correlation
            matrix, as a fraction of the largest; see the svdCut function. The
            eigenvalues a sample covariance gets wrong are the smallest ones, so
            this is what keeps a correlated fit from over-trusting them on the
            100-config Nx=48/64 caches. Defaults to 1e-3; raise it (1e-2) if
            refits scatter across resamples, 0/None for pure GLS.

    Returns:
        MassFit: .masses, .curves, .svdCut, .correlated.
    """
    curves = bootstrapEnsemble(
        measured, weights=weights,
        reduce=makeGevpReduce(ti=ti, shift=shift, labelIdx=labelIdx),
        numResamples=covResamples or min(numResamples, 2000),
        seed=seed, progress=progress, quantile=quantile)

    #cov is None when too few resamples stayed jointly finite; that is a signal
    #the curves are too noisy to weight with, so fall back to the plain fit
    cov = curves[2] if correlated else None
    if cov is None:
        svdCut = 0.0

    masses = bootstrapEnsemble(
        measured, weights=weights,
        reduce=massReduce(ti=ti, shift=shift, fitT=fitT, withAmp=withAmp,
                          labelIdx=labelIdx, fitForm=fitForm, preTi=preTi,
                          cov=cov, svdCut=svdCut),
        numResamples=numResamples, seed=seed, progress=progress, quantile=quantile)

    return MassFit(masses=masses, curves=curves,
                   svdCut=float(svdCut or 0.0), correlated=cov is not None)



