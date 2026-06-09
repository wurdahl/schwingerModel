from __future__ import annotations

import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import splu
from scipy.stats import bootstrap
from scipy.optimize import curve_fit
from tqdm import tqdm

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .schwingerModel import schwingerModel

from .buildOps import buildDiracOp

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
    H-= 2*sparse.identity(modelObj.dimx)

    return H

def findPartialEigenBasis(modelObj: schwingerModel, configIndex = 0, numVecs = 4):

    eigenBases = []

    for nt in range(modelObj.dimt):
        lap = -buildLaplacian(modelObj, modelObj.linkHistory[configIndex], nt=nt)

        #This should find the smallest eigenvalues/eigenvectros of the laplacian
        eigs, eigVecs = sparse.linalg.eigsh(lap, k=numVecs,sigma=0, which='LM')

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
        tau[:, t_src] = np.einsum('tai, atjkd -> tijkd', eigVecs.conj(), Phi).reshape(N_t, N_vec*2, N_vec*2)

    return tau

def getCorrelation(modelObj: schwingerModel, configIndex: int, numVecs: int, chemicalPot=0,
                    gamma=np.array([[1j,0],[0,-1j]])):

    peramb = buildPerambulator(modelObj,configIndex, numVecs, chemicalPot)

    elemental = np.kron(np.eye(numVecs),gamma)

    trace = -np.einsum("ijkl,lm,jimn,nk->ij",peramb,elemental,peramb,elemental,optimize=True)

    correlator = np.array([
        np.roll(trace, -dt, axis=0).diagonal().mean()
        for dt in range(modelObj.dimt)
    ]).real

    return correlator