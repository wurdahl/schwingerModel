from __future__ import annotations

import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import splu
from scipy.stats import bootstrap
from scipy.optimize import curve_fit
from tqdm.auto import tqdm
from joblib import Parallel, delayed
import joblib
import contextlib

@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    class _Callback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)
    old = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = _Callback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old
        tqdm_object.close()

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .schwingerModel import schwingerModel

from .buildOps import buildDiracOp
from .analysis import getWeightingFactorsTheta
from . import wick

def buildLaplacian(modelObj: schwingerModel, gaugeLinks, nt):
    """
    Creates the gauge-covariant laplacian at time slice nt (no spin index)
    """

    #dirac dimensions

    shift_x_1Dpos = np.roll(np.eye(modelObj.dimx), -1, axis=0) # This is \delta_{x_n+1, x_m}
    shift_x_1Dneg = np.roll(np.eye(modelObj.dimx), +1, axis=0) # This is \delta_{x_n+1, x_m}

    #flattened gaugelinks: [:,nt, 1] are spatial links at timeslice t
    spaceLinks = sparse.diags_array(gaugeLinks[:,nt,1].flatten())

    #H matrix for smearing
    H = spaceLinks@shift_x_1Dpos + shift_x_1Dneg@np.conj(spaceLinks)
    #subtract off diagonal
    H-= 2*sparse.eye_array(modelObj.dimx)

    return H

def findPartialEigenBasis(modelObj: schwingerModel, configIndex = 0, numVecs = 4):

    eigenBases = []

    for nt in range(modelObj.dimt):
        lap = -buildLaplacian(modelObj, modelObj.linkHistory[configIndex], nt=nt)

        #This should find the smallest eigenvalues/eigenvectors of the laplacian
        eigs, eigVecs = sparse.linalg.eigsh(lap, k=numVecs,sigma=0, which='LM')

        #momentum projection
        # eigVecs *= np.exp(-1j*2*np.pi*momk*np.arange(modelObj.dimx)/modelObj.dimx)

        eigenBases.append(eigVecs)

    return np.array(eigenBases) #shape: (dimt, dimx, numVecs)

def buildPerambulator(modelObj: schwingerModel, configIndex: int, numVecs: int, chemicalPot=0):
    """
    Computes the distillation perambulator for a single gauge configuration.

    Returns tau of shape (dimt, dimt, numVecs*2, numVecs*2)
      tau[t_sink, t_src, l*2+s_sink, k*2+s_src]
        = sum_x V(t_sink)[x,l]* M^{-1}[x,t_sink,s_sink; x',t_src,s_src] V(t_src)[x',k]
    """
    gaugeLinks = modelObj.linkHistory[configIndex]
    eigVecs = findPartialEigenBasis(modelObj, configIndex, numVecs)

    # eigVecs shape: (dimt, dimx, numVecs)

    N_t, N_x, N_vec = eigVecs.shape

    lu = splu(buildDiracOp(modelObj, gaugeLinks, chemicalPot).tocsc())

    tau = np.zeros((N_t, N_t, N_vec*2, N_vec*2), dtype=complex)

    for t_src in range(N_t):
        # Build sources: one column per (k, s), localized at t_src
        B = np.zeros((N_x*N_t*2, N_vec*2), dtype=complex)
        for s in range(2):
            rows = np.arange(N_x)*N_t*2 + t_src*2 + s
            B[np.ix_(rows, np.arange(N_vec)*2 + s)] = eigVecs[t_src]  # (N_x, N_vec)

        Phi = lu.solve(B).reshape(N_x, N_t, 2, N_vec, 2)
        # (x, t_sink, s_sink, k_src, s_src)

        # einsum: t=t_sink, a=x (contracted), i=l_sink, j=s_sink, k=k_src, d=s_src
        # compound row index: l*2+s_sink, compound col index: k*2+s_src
        tau[:, t_src] = np.einsum('tai, atjkd -> tijkd', eigVecs.conj(), Phi, optimize=True).reshape(N_t, N_vec*2, N_vec*2)

    return tau

def covariantDerivative(modelObj: schwingerModel, gaugeLinks, nt):
    """
    Symmetric gauge-covariant spatial derivative at time slice nt (no spin index):
        (D psi)(x) = [U_x(x) psi(x+1) - U_x^*(x-1) psi(x-1)] / (2a)
    """
    shift_x_1Dpos = np.roll(np.eye(modelObj.dimx), -1, axis=0) # This is \delta_{x_n+1, x_m}
    shift_x_1Dneg = np.roll(np.eye(modelObj.dimx), +1, axis=0) # This is \delta_{x_n-1, x_m}

    #flattened gaugelinks: [:,nt, 1] are spatial links at timeslice t
    spaceLinks = sparse.diags_array(gaugeLinks[:,nt,1].flatten())

    D = (spaceLinks@shift_x_1Dpos - shift_x_1Dneg@(spaceLinks.conj()))/(2*modelObj.a)

    return D

def derivativeKernel(numDerivs):
    """
    Returns a spatial kernel function K(modelObj, gaugeLinks, nt) = D^numDerivs
    for use in buildElemental. numDerivs=0 gives the identity.
    NOTE: odd numbers of derivatives flip the spatial parity of the operator, so at
    zero momentum their cross-correlators with even-derivative operators vanish.
    """
    def kernel(modelObj, gaugeLinks, nt):
        K = sparse.eye_array(modelObj.dimx)
        for _ in range(numDerivs):
            K = covariantDerivative(modelObj, gaugeLinks, nt) @ K
        return K
    return kernel

def buildElemental(modelObj: schwingerModel, configIndex:int, numVecs: int, kernel=None,
                    gamma=np.array([[1j,0],[0,-1j]]), eigVecs=None):
    """
    Builds the meson elemental for one configuration:
        Phi[t] = kron( V(t)^dag K(t) V(t), gamma )
    shape (dimt, numVecs*2, numVecs*2) with the compound index l*2+s matching
    buildPerambulator. kernel=None means the identity spatial kernel, for which
    orthonormality of V gives Phi[t] = kron(eye(numVecs), gamma) at every t.
    """
    gaugeLinks = modelObj.linkHistory[configIndex]
    if eigVecs is None:
        eigVecs = findPartialEigenBasis(modelObj, configIndex, numVecs)

    N_t, N_x, N_vec = eigVecs.shape

    elemental = np.zeros((N_t, N_vec*2, N_vec*2), dtype=complex)
    for nt in range(N_t):
        if kernel is None:
            vkv = eigVecs[nt].conj().T @ eigVecs[nt]
        else:
            vkv = eigVecs[nt].conj().T @ (kernel(modelObj, gaugeLinks, nt) @ eigVecs[nt])
        elemental[nt] = np.kron(vkv, gamma)

    return elemental

def operatorBasis(maxDerivs=2, gammas=None):
    """
    Standard operator basis for GEVP: every gamma structure paired with
    0..maxDerivs covariant derivatives. gammas is a dict {name: (2,2) array},
    defaulting to the pseudoscalar used everywhere else in this repo.
    """
    if gammas is None:
        gammas = {"g5": np.array([[1j,0],[0,-1j]])}

    ops = []
    for gName, gamma in gammas.items():
        for nD in range(maxDerivs+1):
            kernel = derivativeKernel(nD) if nD > 0 else None
            ops.append(wick.mesonOp(f"{gName}_D{nD}", gamma, kernel))
    return ops

class distillationSpace:
    """
    Manages the distillation objects for one model: eigenvector bases,
    perambulators, and elementals are computed lazily and cached per
    configuration (and per operator for the elementals).
    """

    def __init__(self, modelObj: schwingerModel, numVecs: int, chemicalPot=0):
        self.modelObj = modelObj
        self.numVecs = numVecs
        self.chemicalPot = chemicalPot

        self._eigenBases = {}
        self._perambulators = {}
        self._elementals = {}   # keyed by (configIndex, op.name)

    def eigenBasis(self, configIndex):
        if configIndex not in self._eigenBases:
            self._eigenBases[configIndex] = findPartialEigenBasis(self.modelObj, configIndex, self.numVecs)
        return self._eigenBases[configIndex]

    def perambulator(self, configIndex):
        if configIndex not in self._perambulators:
            self._perambulators[configIndex] = buildPerambulator(self.modelObj, configIndex,
                                                                 self.numVecs, chemicalPot=self.chemicalPot)
        return self._perambulators[configIndex]

    def elemental(self, configIndex, op: wick.mesonOp):
        if (configIndex, op.name) not in self._elementals:
            self._elementals[(configIndex, op.name)] = buildElemental(
                self.modelObj, configIndex, self.numVecs, kernel=op.kernel,
                gamma=op.gamma, eigVecs=self.eigenBasis(configIndex))
        return self._elementals[(configIndex, op.name)]

    def elementalBar(self, configIndex, op: wick.mesonOp):
        """
        Elemental of the daggered operator: (psibar G K psi)^dag = psibar Gbar K^dag psi
        with Gbar = gammat G^dag gammat, so PhiBar[t] = Phi[t]^dag conjugated by gammat.
        """
        gt = np.kron(np.eye(self.numVecs), self.modelObj.gammat)
        phi = self.elemental(configIndex, op)
        return gt @ phi.conj().transpose(0,2,1) @ gt

def getElementalCorrelMatrix(modelObj: schwingerModel, configIndex: int, numVecs: int, ops,
                              chemicalPot=0):
    """
    Per-configuration pieces of the correlation matrix C_ij(t) = <O_i(t) O_j(0)^dag>
    for a list of mesonOps (flavor coefficients are applied later by the driver).

    Returns (conn, loopsSnk, loopsSrc):
        conn[i,j,dt]  = (1/T) sum_t tr[Phi_i(t+dt) tau(t+dt,t) PhiBar_j(t) tau(t,t+dt)]
        loopsSnk[i,t] = tr[Phi_i(t) tau(t,t)]
        loopsSrc[j,t] = tr[PhiBar_j(t) tau(t,t)]
    """
    space = distillationSpace(modelObj, numVecs, chemicalPot=chemicalPot)
    tau = space.perambulator(configIndex)
    tauDiag = np.einsum('ttij->tij', tau)

    nOps = len(ops)
    dimt = modelObj.dimt

    conn = np.zeros((nOps, nOps, dimt), dtype=complex)
    loopsSnk = np.zeros((nOps, dimt), dtype=complex)
    loopsSrc = np.zeros((nOps, dimt), dtype=complex)

    for i, opSnk in enumerate(ops):
        phiSnk = space.elemental(configIndex, opSnk)
        loopsSnk[i] = np.einsum('tab,tba->t', phiSnk, tauDiag, optimize=True)

        for j, opSrc in enumerate(ops):
            phiBarSrc = space.elementalBar(configIndex, opSrc)
            if i == 0:
                loopsSrc[j] = np.einsum('tab,tba->t', phiBarSrc, tauDiag, optimize=True)

            # trace[t_snk, t_src] = tr[Phi_i(t_snk) tau(t_snk,t_src) PhiBar_j(t_src) tau(t_src,t_snk)]
            trace = np.einsum('tab,tsbc,scd,stda->ts', phiSnk, tau, phiBarSrc, tau, optimize=True)

            #average over source position at fixed separation
            conn[i,j] = np.array([
                np.roll(trace, -dt, axis=0).diagonal().mean()
                for dt in range(dimt)
            ])

    return conn, loopsSnk, loopsSrc

def getCorrelation(modelObj: schwingerModel, configIndex: int, numVecs: int, chemicalPot=0,
                    gamma=np.array([[1j,0],[0,-1j]])):

    peramb = buildPerambulator(modelObj, configIndex, numVecs, chemicalPot=chemicalPot)

    elemental = np.kron(np.eye(numVecs),gamma)

    trace = -np.einsum("ijkl,lm,jimn,nk->ij",peramb,elemental,peramb,elemental,optimize=True)

    correlator = np.array([
        np.roll(trace, -dt, axis=0).diagonal().mean()
        for dt in range(modelObj.dimt)
    ])

    return correlator

def getCorrelationLoop(modelObj: schwingerModel, configIndex: int, numVecs: int, chemicalPot=0,
                    gamma=np.array([[1j,0],[0,-1j]])):
    #assuming isospin symmetry for everything
    
    peramb = buildPerambulator(modelObj, configIndex, numVecs, chemicalPot=chemicalPot)

    elemental = np.kron(np.eye(numVecs),gamma)

    trace = np.einsum("iikl,lk->i",peramb,elemental,optimize=True)

    return trace

def correlStats(modelObj: schwingerModel, burnIn=1, autocorrSkip=1,
                    Gamma=np.array([[1j,0],[0,-1j]]), nVec=2, chemicalPot=0, theta=0,disc=False):
    
    weights = getWeightingFactorsTheta(modelObj, theta=theta,burnIn=burnIn, autocorrSkip=autocorrSkip)
    
    indices = np.arange(burnIn, modelObj.metroSteps, autocorrSkip)
    with tqdm_joblib(tqdm(total=len(indices), desc="Conn. configs")):
        correl = np.array(Parallel(n_jobs=-1)(delayed(getCorrelation)(modelObj, i, nVec, gamma=Gamma,chemicalPot=chemicalPot) for i in indices))

    if(disc):
        with tqdm_joblib(tqdm(total=len(indices), desc="Disc. Loops")):
            loops = np.array(Parallel(n_jobs=-1)(delayed(getCorrelationLoop)(modelObj, i, nVec, gamma=Gamma,chemicalPot=chemicalPot) for i in indices))
        # loops shape: (n_configs, dimt), loops[n,t] = L_n(t) = Tr[Phi tau(t,t)]

    totalCorrelMean = np.real(np.average(correl,axis=0,weights=weights))

    if(disc):
        dimt = modelObj.dimt

        # per-config, translation-averaged loop-loop product on the SAME config:
        #   loopCorrel[n, dt] = (1/T) sum_t L_n(t+dt) L_n(t)
        loopCorrel = np.stack([
            np.mean(np.roll(loops, -dt, axis=1) * loops, axis=1)
            for dt in range(dimt)
        ], axis=1)                                       # (n_configs, dimt)

        def _discCorrel(llc, lp, w):
            # llc, lp: (..., n_configs, dimt); w: (..., n_configs)
            # returns vacuum-subtracted disconnected correlator (..., dimt)
            ws   = np.sum(w, axis=-1, keepdims=True)
            LL   = np.sum(llc * w[..., None], axis=-2) / ws          # <(1/T) sum_t L(t+dt)L(t)>
            Lbar = np.sum(lp  * w[..., None], axis=-2) / ws          # <L(t)>  (..., dimt)
            vac  = np.stack([                                        # (1/T) sum_t <L(t+dt)><L(t)>
                np.mean(np.roll(Lbar, -dt, axis=-1) * Lbar, axis=-1)
                for dt in range(dimt)
            ], axis=-1)
            return LL - vac

        totalCorrelMean = totalCorrelMean + np.real(_discCorrel(loopCorrel, loops, weights))

    # Chunked bootstrap: never materialise (numResamples, n_configs, dimt) all at
    # once — instead process in small batches and accumulate the per-sample means.
    numResamples = 10000
    chunk_size   = 500
    rng = np.random.default_rng()

    bootstrap_means = np.empty((numResamples, modelObj.dimt))

    for start in range(0, numResamples, chunk_size):
        end     = min(start + chunk_size, numResamples)
        idx     = rng.choice(len(correl), size=(end - start, len(correl)))
        w_chunk = weights[idx]                           # (chunk, n_configs)

        correl_chunk = correl[idx]                       # (chunk, n_configs, dimt)
        chunk_means  = np.real(
            np.sum(correl_chunk * w_chunk[:, :, np.newaxis], axis=1) /
            np.sum(w_chunk, axis=1, keepdims=True)
        )
        del correl_chunk

        if(disc):
            chunk_means = chunk_means + np.real(
                _discCorrel(loopCorrel[idx], loops[idx], w_chunk)
            )

        bootstrap_means[start:end] = chunk_means

    low  = np.percentile(bootstrap_means, 2.5,  axis=0)
    high = np.percentile(bootstrap_means, 97.5, axis=0)

    covMat = np.real(np.cov(bootstrap_means,rowvar=False))

    return [totalCorrelMean, np.array([high-totalCorrelMean, totalCorrelMean-low]).real, covMat]


def correlMassExtract(correlStatsOut, fitT=[1,10],diagCov=False):
    """
    correlMassExtract: given an output of correlStatsOut, fit a cosh in log space to determine mass of particle

    fitT - time slices to fit the cosh expreession to. correlation will be divided by fitT[0] in order to normalize
    """

    dimt=correlStatsOut[0].shape[0]

    def coshCorrel_log(nt, Energy):
        numer = np.logaddexp(-nt * Energy, (nt - dimt) * Energy)
        denom = np.logaddexp(-fitT[0] * Energy, (fitT[0] - dimt) * Energy)
        return numer - denom
    
    mean = correlStatsOut[0][fitT[0]+1:fitT[1]]
    cov = correlStatsOut[2][fitT[0]+1:fitT[1],fitT[0]+1:fitT[1]]

    log_mean = np.log(mean) - np.log(correlStatsOut[0][fitT[0]])
    inv_mean = 1.0/mean
    log_cov = cov*np.outer(inv_mean,inv_mean)


    if(not diagCov):
        fitMass = curve_fit(coshCorrel_log, xdata=np.arange(fitT[0]+1, fitT[1]),
                    ydata=log_mean, sigma=log_cov, absolute_sigma=True, bounds=(0, np.inf))
    else:
        fitMass = curve_fit(coshCorrel_log, xdata=np.arange(fitT[0]+1, fitT[1]),
                    ydata=log_mean, sigma=np.sqrt(np.diag(log_cov)), absolute_sigma=True, bounds=(0, np.inf))
    
    return np.array([fitMass[0][0], np.sqrt(fitMass[1][0, 0])])


def _discCorrelPair(loopCorrel, loopsSnk, loopsSrc, w):
    """
    Vacuum-subtracted disconnected correlator from two (possibly different) loops.
    loopCorrel: (..., n_configs, dimt) per-config translation-averaged LsnkLsrc product
    loopsSnk, loopsSrc: (..., n_configs, dimt); w: (..., n_configs)
    Returns (..., dimt).
    """
    dimt = loopsSnk.shape[-1]
    ws    = np.sum(w, axis=-1, keepdims=True)
    LL    = np.sum(loopCorrel * w[..., None], axis=-2) / ws     # <(1/T) sum_t Lsnk(t+dt)Lsrc(t)>
    LbarS = np.sum(loopsSnk   * w[..., None], axis=-2) / ws     # <Lsnk(t)>
    LbarB = np.sum(loopsSrc   * w[..., None], axis=-2) / ws     # <Lsrc(t)>
    vac   = np.stack([                                          # (1/T) sum_t <Lsnk(t+dt)><Lsrc(t)>
        np.mean(np.roll(LbarS, -dt, axis=-1) * LbarB, axis=-1)
        for dt in range(dimt)
    ], axis=-1)
    return LL - vac


def distillGEVPStats(modelObj: schwingerModel, ops=None, flavorTerms=wick.PION_PLUS,
                     burnIn=1, autocorrSkip=1, numVecs=4, chemicalPot=0, theta=0,
                     ti=1, numResamples=2000, n_jobs=-1):
    """
    Distillation GEVP over a basis of mesonOps (different gammas / covariant
    derivatives): builds the full correlation matrix C_ij(t) = <O_i(t) O_j(0)^dag>
    including the disconnected diagrams required by the channel's flavor structure
    (wick.PION_PLUS / PION_ZERO: connected only, wick.ETA: connected + 2x disc.),
    then solves the GEVP on the ensemble mean and on each bootstrap resample.

    Returns [mean (dimt-ti, nOps), errors (2, dimt-ti, nOps), covMat (nOps, dimt-ti, dimt-ti)],
    compatible with correlation.gevpMassExtract.
    """
    from .correlation import gevp

    if ops is None:
        ops = operatorBasis(maxDerivs=2)
    nOps = len(ops)
    dimt = modelObj.dimt

    #wick contraction coefficients from the permutation method
    connCoeff, discCoeff = wick.twoPointCoeffs(flavorTerms)

    weights = getWeightingFactorsTheta(modelObj, theta=theta, burnIn=burnIn, autocorrSkip=autocorrSkip)
    indices = np.arange(burnIn, modelObj.metroSteps, autocorrSkip)

    with tqdm_joblib(tqdm(total=len(indices), desc="Distill. configs")):
        perConfig = Parallel(n_jobs=n_jobs)(
            delayed(getElementalCorrelMatrix)(modelObj, i, numVecs, ops, chemicalPot=chemicalPot)
            for i in indices)

    conn     = connCoeff * np.array([r[0] for r in perConfig])   # (n_configs, nOps, nOps, dimt)
    loopsSnk = np.array([r[1] for r in perConfig])               # (n_configs, nOps, dimt)
    loopsSrc = np.array([r[2] for r in perConfig])

    disc = (discCoeff != 0)
    if disc:
        # per-config translation-averaged loop-loop product on the SAME config:
        #   loopCorrel[n, i, j, dt] = (1/T) sum_t Lsnk_i(t+dt) Lsrc_j(t)
        loopCorrel = np.stack([
            np.einsum('nit,njt->nij', np.roll(loopsSnk, -dt, axis=-1), loopsSrc, optimize=True)/dimt
            for dt in range(dimt)
        ], axis=-1)                                              # (n_configs, nOps, nOps, dimt)

    def _meanMatrix(idx=None):
        #weighted-mean correlation matrix; idx indexes bootstrap resamples
        if idx is None:
            c, w = conn, weights
            lC = loopCorrel if disc else None
            lS, lB = (loopsSnk, loopsSrc) if disc else (None, None)
        else:
            c, w = conn[idx], weights[idx]
            lC = loopCorrel[idx] if disc else None
            lS, lB = (loopsSnk[idx], loopsSrc[idx]) if disc else (None, None)

        wExp = w[..., None, None, None]
        C = np.sum(c * wExp, axis=-4) / np.sum(wExp, axis=-4)

        if disc:
            #disc piece for every (i,j): loop over ops (nOps is small)
            for i in range(nOps):
                for j in range(nOps):
                    C[..., i, j, :] = C[..., i, j, :] + discCoeff * _discCorrelPair(
                        lC[..., :, i, j, :], lS[..., :, i, :], lB[..., :, j, :], w)
        return C

    def _gevpEigs(C):
        # C: (nOps, nOps, dimt) -> sorted eigenvalue correlators (dimt-ti, nOps)
        corrMat = np.real((C + np.conj(np.swapaxes(C, 0, 1)))/2)  #hermitize
        newCorrs, _ = gevp(corrMat, ti=ti)
        return np.real(newCorrs)

    totalCorrelMean = _gevpEigs(_meanMatrix())                    # (dimt-ti, nOps)

    #chunked bootstrap over configurations
    chunk_size = 500
    rng = np.random.default_rng()

    bootstrap_means = np.empty((numResamples, dimt - ti, nOps))

    for start in range(0, numResamples, chunk_size):
        end = min(start + chunk_size, numResamples)
        idx = rng.choice(len(conn), size=(end - start, len(conn)))

        C_chunk = _meanMatrix(idx)                                # (chunk, nOps, nOps, dimt)
        for s in range(end - start):
            bootstrap_means[start + s] = _gevpEigs(C_chunk[s])
        del C_chunk

    low  = np.percentile(bootstrap_means, 2.5,  axis=0)           # (dimt-ti, nOps)
    high = np.percentile(bootstrap_means, 97.5, axis=0)

    # Per-eigenvalue covariance: (nOps, dimt-ti, dimt-ti)
    covMat = np.array([np.cov(bootstrap_means[:, :, eigIdx], rowvar=False) for eigIdx in range(nOps)])

    return [totalCorrelMean, np.array([high - totalCorrelMean, totalCorrelMean - low]), covMat]


