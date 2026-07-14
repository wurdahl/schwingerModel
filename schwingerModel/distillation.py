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


