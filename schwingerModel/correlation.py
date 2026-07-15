import numpy as np
import scipy.sparse as sparse
from scipy.stats import bootstrap
from scipy.optimize import curve_fit
from tqdm import tqdm
from scipy.linalg import eig
from scipy.sparse.linalg import splu
from joblib import Parallel, delayed
import joblib
from contextlib import contextmanager

from .schwingerModel import schwingerModel
from . import buildOps as ops
from . import analysis


@contextmanager
def _tqdm_joblib(tqdm_obj):
    """Hook a tqdm bar into joblib.Parallel progress."""
    class _Callback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_obj.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)
    old_cb = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = _Callback
    try:
        yield tqdm_obj
    finally:
        joblib.parallel.BatchCompletionCallBack = old_cb
        tqdm_obj.close()

#calculate the cross correlation between two operators
#there can be different amounts of smearing for the two operators
#this will allow for GEVP analysis
#NOT the most efficient way - written this way to be closest to equations
def getCorrelation(modelObj: schwingerModel, gaugleLinkIndex,
                   smearingOp1=None, smearingOp2=None,
                    Gamma1=np.array([[1j,0],[0,-1j]]), Gamma2=np.array([[1j,0],[0,-1j]]),
                    momk=0, saveProps=True):

    gaugeLinks = modelObj.linkHistory[gaugleLinkIndex]

    # Cache LU factorization of dOp (not the full dense inverse)
    if(hasattr(modelObj, "storedProps") and saveProps):
        if((modelObj.storedProps[gaugleLinkIndex] is None)):
            dOp = ops.buildDiracOp(modelObj, gaugeLinks)
            lu = splu(dOp.tocsc())
            modelObj.storedProps[gaugleLinkIndex] = lu
        else:
            lu = modelObj.storedProps[gaugleLinkIndex]
    else:
        dOp = ops.buildDiracOp(modelObj, gaugeLinks)
        lu = splu(dOp.tocsc())

    N = modelObj.dimx * modelObj.dimt * 2
    E = np.eye(N, 2, dtype=complex)  # first 2 unit vectors, (N, 2)

    # smearingOp can be a dense/sparse matrix or a (H, kappa, smearN) tuple.
    # H is Hermitian, so S is too: S^T = S^* = conj(S).
    # This means S1_rows^T = S1^T @ E = conj(S1 @ E) = conj(S1_cols) — one apply covers both.
    if isinstance(smearingOp1, tuple):
        H1, kappa1, smearN1 = smearingOp1
        S1_cols = ops.applyJacobi(H1, kappa1, smearN1, E)   # (N, 2): S1 @ e_j
        S1_rows_T = np.conj(S1_cols)                          # (N, 2): S1^T @ e_j = conj(S1 @ e_j)
    else:
        S1 = smearingOp1
        S1_cols = S1[:, 0:2].toarray() if sparse.issparse(S1) else S1[:, 0:2]
        S1_rows = S1[0:2, :].toarray() if sparse.issparse(S1) else S1[0:2, :]
        S1_rows_T = S1_rows.T

    X = lu.solve(S1_cols)           # D^{-1} S1[:,0:2], (N, 2)
    Y = lu.solve(S1_rows_T, trans='T')  # D^{-T} S1_rows^T, (N, 2)

    if isinstance(smearingOp2, tuple):
        H2, kappa2, smearN2 = smearingOp2
        # Y^T @ S2 = (S2^T @ Y)^T = conj(S2 @ conj(Y))^T  (using S2^T = S2^*)
        prop12_partial = np.conj(ops.applyJacobi(H2, kappa2, smearN2, np.conj(Y))).T  # (2, N)
        prop21_partial = ops.applyJacobi(H2, kappa2, smearN2, X)                       # (N, 2)
    else:
        S2 = smearingOp2
        if sparse.issparse(S2):
            prop12_partial = (S2.T @ Y).T
        else:
            prop12_partial = Y.T @ S2
        prop21_partial = S2 @ X

    stridex = modelObj.dimt*2
    stridet = 2

    correl_conn = np.zeros(modelObj.dimt,dtype=np.complex128)

    for x_m in range(modelObj.dimx):
        momPhase = np.exp(-1j*2*np.pi*momk*x_m/modelObj.dimx)
        for t_m in range(modelObj.dimt):
            idx_m_start = x_m * stridex + t_m * stridet
            idx_m_end = idx_m_start + 2

            propnm_12 = prop12_partial[:, idx_m_start:idx_m_end]   # (2, 2)
            propmn_21 = prop21_partial[idx_m_start:idx_m_end, :]   # (2, 2)

            correl_conn[t_m] += -np.trace(Gamma1@propnm_12@Gamma2@propmn_21)*momPhase

    return np.real(correl_conn)

def correlStats(modelObj: schwingerModel, burnIn=1, autocorrSkip=1,
                    Gamma1=np.array([[1j,0],[0,-1j]]), Gamma2=None,
                    kappa1=0, smearN1=0, kappa2=0, smearN2=0,
                    momk=0, saveProps=True):
    if(Gamma2 is None):
        Gamma2 = Gamma1
    
    acceptedCorrel_conn = []

    #weights for chemicalPot
    weightsMu = analysis.getWeightingFactors(modelObj, 0, burnIn,  autocorrSkip)

    #if k!=0, then each config will show up twice, so we need to repeat the weights
    if(momk!=0):
        weightsMu = np.repeat(weightsMu,2)

    for i in tqdm(range(burnIn,modelObj.metroSteps,autocorrSkip)):
        jacobiS1 = ops.jacobiSmearingOp(modelObj, modelObj.linkHistory[i],kappa1,smearN1)
        jacobiS2 = ops.jacobiSmearingOp(modelObj, modelObj.linkHistory[i],kappa2,smearN2)
        Cconn = getCorrelation(modelObj, i, 
                               smearingOp1=jacobiS1,
                               smearingOp2=jacobiS2,
                               Gamma1=Gamma1, Gamma2=Gamma2, momk=momk, saveProps=saveProps)
        
        acceptedCorrel_conn.append(Cconn)

        if(momk!=0):
            Cconn = getCorrelation(modelObj, i, 
                               smearingOp1=jacobiS1,
                               smearingOp2=jacobiS2,
                               Gamma1=Gamma1, Gamma2=Gamma2, momk=-momk, saveProps=saveProps)
        
            acceptedCorrel_conn.append(Cconn)


    acceptedCorrel_conn = np.array(acceptedCorrel_conn)

    totalCorrels = acceptedCorrel_conn

    totalCorrelMean = np.real(np.average(totalCorrels,axis=0,weights=weightsMu))

    #bootstrapping
    numResamples = 10000
    rng = np.random.default_rng()

    resamples = rng.choice(len(totalCorrels), size=(numResamples, len(totalCorrels)))

    # (numResamples, n_configs, dimt) and (numResamples, n_configs)
    correl_boot = totalCorrels[resamples]
    w_boot = weightsMu[resamples]

    # weighted mean for each bootstrap sample -> (numResamples, dimt)
    bootstrap_means = np.real(
        np.sum(correl_boot * w_boot[:, :, np.newaxis], axis=1) /
        np.sum(w_boot, axis=1, keepdims=True)
    )

    low  = np.percentile(bootstrap_means, 2.5,  axis=0)
    high = np.percentile(bootstrap_means, 97.5, axis=0)

    return [totalCorrelMean, np.array([high-totalCorrelMean, totalCorrelMean-low])]

def _gevp_one_config(modelObj, config_idx, kappas, smearNs,
                     Gamma, momk, saveProps):
    """Per-config work for GEVPStats.

    Returns a flat list of n*(n+1)/2 correlators per momentum mode, ordered as
    (0,0), (0,1), ..., (0,n-1), (1,1), ..., (n-1,n-1).
    If momk != 0, the +k block comes first, then the -k block.
    """
    H = ops.jacobiSmearingH(modelObj, modelObj.linkHistory[config_idx]).tocsc()
    smearingOps = [(H, kappa, smearN) for kappa, smearN in zip(kappas, smearNs)]

    n = len(smearingOps)
    out = []
    ks = [momk, -momk] if momk != 0 else [momk]

    for k in ks:
        for i in range(n):
            for j in range(i, n):
                correl = getCorrelation(modelObj, config_idx, smearingOps[i], smearingOps[j],
                                        Gamma1=Gamma, Gamma2=Gamma, momk=k, saveProps=saveProps)
                out.append(correl)

    return out


def GEVPStats(modelObj: schwingerModel, burnIn=1, autocorrSkip=1,
                    Gamma=np.array([[1j,0],[0,-1j]]),
                    kappas=[.1,0], smearNs=[1,0],
                    momk=0, ti=1, saveProps=True, n_jobs=-1):

    weightsMu = analysis.getWeightingFactors(modelObj, 0, burnIn, autocorrSkip)

    if momk != 0:
        weightsMu = np.repeat(weightsMu, 2)

    indices = list(range(burnIn, modelObj.metroSteps, autocorrSkip))

    with _tqdm_joblib(tqdm(total=len(indices))):
        per_config = Parallel(n_jobs=n_jobs, backend='threading')(
            delayed(_gevp_one_config)(
                modelObj, i, kappas, smearNs,
                Gamma, momk, saveProps
            ) for i in indices
        )

    n = len(kappas)
    n_pairs = n * (n + 1) // 2
    pairs = [(i, j) for i in range(n) for j in range(i, n)]

    # Split each config's flat list into per-momentum blocks: (n_samples, n_pairs, dimt)
    all_samples = []
    for cfg_result in per_config:
        if momk != 0:
            all_samples.append(cfg_result[:n_pairs])
            all_samples.append(cfg_result[n_pairs:])
        else:
            all_samples.append(cfg_result)
    all_samples = np.array(all_samples)
    dimt = all_samples.shape[2]

    # Run GEVP per sample, keeping all n eigenvalues: (n_samples, dimt-ti, n)
    totalCorrels = np.zeros((len(all_samples), dimt - ti, n))
    for s_idx in range(len(all_samples)):
        corrMat = np.zeros((n, n, dimt))
        for p_idx, (oi, oj) in enumerate(pairs):
            corrMat[oi, oj] = all_samples[s_idx, p_idx]
            corrMat[oj, oi] = all_samples[s_idx, p_idx]
        newCorrs, _ = gevp(corrMat, ti=ti)
        totalCorrels[s_idx] = np.real(newCorrs)

    # Bootstrapping
    numResamples = 10000
    rng = np.random.default_rng()

    resamples = rng.choice(len(totalCorrels), size=(numResamples, len(totalCorrels)))

    # correl_boot: (numResamples, n_samples, dimt-ti, n)
    correl_boot = totalCorrels[resamples]
    w_boot = weightsMu[resamples]  # (numResamples, n_samples)

    # weighted mean -> (numResamples, dimt-ti, n)
    numerator = np.sum(correl_boot * w_boot[:, :, np.newaxis, np.newaxis], axis=1)
    denominator = np.sum(w_boot, axis=1)[:, np.newaxis, np.newaxis]
    bootstrap_means = np.real(numerator / denominator)

    # Per-eigenvalue covariance: (n, dimt-ti, dimt-ti)
    covMat = np.array([np.cov(bootstrap_means[:, :, eig_idx], rowvar=False) for eig_idx in range(n)])

    low  = np.percentile(bootstrap_means, 2.5,  axis=0)  # (dimt-ti, n)
    high = np.percentile(bootstrap_means, 97.5, axis=0)  # (dimt-ti, n)

    totalCorrelMean = np.real(np.average(totalCorrels, axis=0, weights=weightsMu))  # (dimt-ti, n)

    return [totalCorrelMean, np.array([high - totalCorrelMean, totalCorrelMean - low]), covMat]

    
def gevp(corrMat, ti=1, refT=None):
    """
    corrMat: (n, n, dimt) symmetric correlation matrix.
    Solves C(t) v = lambda C(ti) v for every t >= ti.

    States are labelled by their EIGENVECTORS, not by per-timeslice magnitude
    sorting: a reference solve at refT (default ti+1) defines the states (sorted
    descending there), and at every other t each eigenvalue is assigned to the
    state whose reference eigenvector it overlaps most in the C(ti) metric.
    This keeps a state's correlator on one curve even when eigenvalue curves
    cross (e.g. backward cosh branches or noise at large t).

    Returns newCorr (dimt-ti, n) and the reference eigenvectors (n, n) whose
    columns give each state's operator content.
    """
    from scipy.optimize import linear_sum_assignment

    n = corrMat.shape[0]
    dimt = corrMat.shape[2]
    ref = corrMat[:, :, ti]
    if refT is None:
        refT = ti + 1

    def _normalizedEig(t):
        evals, evecs = eig(a=corrMat[:, :, t], b=ref)
        #a near-singular reference produces inf/nan eigenpairs -- flag them so
        #they never win the overlap assignment or reach the fits as huge numbers
        bad = ~np.isfinite(evals)
        evals = np.where(bad, np.nan, evals)
        evecs[:, bad] = 0
        #normalize in the C(ti) metric so overlaps are comparable
        norms = np.sqrt(np.abs(np.einsum('ai,ab,bi->i', evecs.conj(), ref, evecs)))
        norms[~np.isfinite(norms) | (norms == 0)] = np.inf
        return evals, evecs/norms

    #reference solve defines the state labels
    evalsR, evecsR = _normalizedEig(refT)
    order = np.argsort(np.nan_to_num(np.real(evalsR), nan=-np.inf))[::-1]
    evecsR = evecsR[:, order]

    newCorr = np.zeros((dimt - ti, n))
    for t in range(ti, dimt):
        evals, evecs = _normalizedEig(t)
        #assign each state the eigenvalue whose eigenvector it overlaps most
        overlap = np.abs(evecsR.conj().T @ ref @ evecs)   # (state, eigenvalue)
        overlap = np.nan_to_num(overlap, nan=0.0, posinf=0.0, neginf=0.0)
        _, col = linear_sum_assignment(-overlap)
        newCorr[t - ti] = np.real(evals)[col]

    return newCorr, evecsR

def gevpMassExtract(gevpStatsOut, fitT=[1,10], ti=1, eigenIdx=0, coshExpr=True):
    """
    gevpStatsOut: output of GEVPStats — [mean (dimt-ti, n), errors, covMat (n, dimt-ti, dimt-ti)]
    eigenIdx: which eigenvalue to fit (0 = lowest mass, 1 = next, ...)

    Fits in log space: minimizes relative residuals, giving equal weight per decade.
    Covariance is propagated as Σ_log[i,j] = Σ_lin[i,j] / (C[i] * C[j]).
    """
    dimt = gevpStatsOut[0].shape[0] + ti

    def expDecay_log(nt, Energy):
        return -nt * Energy

    def coshCorrel_log(nt, Energy):
        numer = np.logaddexp(-(nt + ti) * Energy, ((nt + ti) - dimt) * Energy)
        denom = np.logaddexp(-ti * Energy, (ti - dimt) * Energy)
        return numer - denom

    mean = gevpStatsOut[0][fitT[0]:fitT[1], eigenIdx]
    cov  = gevpStatsOut[2][eigenIdx, fitT[0]:fitT[1], fitT[0]:fitT[1]]

    log_mean = np.log(mean)
    inv_mean = 1.0 / mean
    log_cov  = cov * np.outer(inv_mean, inv_mean)

    if coshExpr:
        fitMass = curve_fit(coshCorrel_log, xdata=np.arange(fitT[0], fitT[1]),
                    ydata=log_mean, sigma=log_cov, absolute_sigma=True, bounds=(0, np.inf))
    else:
        fitMass = curve_fit(expDecay_log, xdata=np.arange(fitT[0], fitT[1]),
                    ydata=log_mean, sigma=log_cov, absolute_sigma=True, bounds=(0, np.inf))

    return np.array([fitMass[0][0], np.sqrt(fitMass[1][0, 0])])
