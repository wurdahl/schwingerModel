from __future__ import annotations

import numpy as np
import scipy.sparse as sparse
from scipy.stats import bootstrap
from scipy.optimize import curve_fit
from tqdm import tqdm

from .params import LatticeParams, dwfParams

#builds the dirac operator using the global gaugeLinks configuration
# matrix is a square matrix with dimensional ordering (space, time, spin) 
def buildDiracOp(modelSettings: LatticeParams, gaugeLinks, chemicalPot=0):
    #dirac dimensions
    dimD = 2
    eyeD = np.eye(dimD)

    shift_x_1Dpos = np.roll(np.eye(modelSettings.dimx), -1, axis=0) # This is \delta_{x_n+1, x_m}
    shift_t_1Dpos = np.roll(np.eye(modelSettings.dimt), -1, axis=0)
    shift_x_1Dneg = np.roll(np.eye(modelSettings.dimx), +1, axis=0) # This is \delta_{x_n-1, x_m}
    shift_t_1Dneg = np.roll(np.eye(modelSettings.dimt), +1, axis=0)
    time_identity = sparse.eye_array(modelSettings.dimt)                      # This is \delta_{t_n, t_m}
    space_identity = sparse.eye_array(modelSettings.dimx)

    #anti-periodic boundary conditions for fermions in time
    shift_t_1Dpos[modelSettings.dimt - 1, 0] = -1.0
    shift_t_1Dneg[0, modelSettings.dimt - 1] = -1.0

    #space-time shift operators
    T_x_pos = sparse.kron(shift_x_1Dpos, time_identity)
    T_x_neg = sparse.kron(shift_x_1Dneg, time_identity)
    T_t_pos = sparse.kron(space_identity, shift_t_1Dpos)
    T_t_neg = sparse.kron(space_identity, shift_t_1Dneg)

    #flattened gaugelinks
    spaceLinks = sparse.diags_array(gaugeLinks[:,:,1].flatten())
    timeLinks = sparse.diags_array(gaugeLinks[:,:,0].flatten())

    #start building dirac operator matrix
    Dee = (modelSettings.fMass+2/modelSettings.a)*sparse.kron(space_identity, sparse.kron(time_identity,eyeD))
    #positive shifts
    Dee-=1/(2*modelSettings.a) * sparse.kron(spaceLinks@T_x_pos, eyeD-modelSettings.gammax)
    Dee-=1/(2*modelSettings.a) * sparse.kron(timeLinks@T_t_pos, eyeD-modelSettings.gammat)*np.exp(modelSettings.a*chemicalPot)
    #negative shifts
    Dee-=1/(2*modelSettings.a) * sparse.kron(T_x_neg@(spaceLinks.conj()),eyeD+modelSettings.gammax)
    Dee-=1/(2*modelSettings.a) * sparse.kron(T_t_neg@(timeLinks.conj()),eyeD+modelSettings.gammat)*np.exp(-modelSettings.a*chemicalPot)

    return Dee

def buildDomainWall5(modelSettings: dwfParams, gaugeLinks):
    Pminus = (np.eye(2) - modelSettings.gamma5)/2
    Pplus  = (np.eye(2) + modelSettings.gamma5)/2

    N5 = modelSettings.dim5

    #identity term
    D5 = sparse.eye_array(N5*modelSettings.dimx*modelSettings.dimt*2)

    #projector terms
    shift5_pos = sparse.eye_array(N5,k=1)
    xt_identity = sparse.eye_array(modelSettings.dimx*modelSettings.dimt)
    D5-= sparse.kron(shift5_pos, sparse.kron(xt_identity, Pminus))

    shift5_neg = sparse.eye_array(N5,k=-1)
    D5-=sparse.kron(shift5_neg, sparse.kron(xt_identity, Pplus))

    #mass terms
    massPos = sparse.csr_array(([1.0], ([N5-1], [0])), shape=(N5, N5))   # \delta_{s,N5-1} \delta_{0,r}
    D5+=modelSettings.fMass * sparse.kron(massPos, sparse.kron(xt_identity,Pminus))

    massNeg = sparse.csr_array(([1.0], ([0], [N5-1])), shape=(N5, N5))   # \delta_{s,0} \delta_{N5-1,r}
    D5+=modelSettings.fMass * sparse.kron(massNeg, sparse.kron(xt_identity,Pplus))

    return D5


def buildDwfOp(modelSettings: dwfParams, gaugeLinks):

    wilsonSettings = LatticeParams(dimx=modelSettings.dimx,dimt=modelSettings.dimt,
                                   beta=modelSettings.beta,fMass=-modelSettings.M5, a=modelSettings.a)

    wilsonOp = buildDiracOp(wilsonSettings, gaugeLinks)

    dim5Id = sparse.eye_array(modelSettings.dim5)

    return sparse.kron(dim5Id, wilsonOp) + buildDomainWall5(modelSettings, gaugeLinks)




def applyCovDerivative(modelSettings: LatticeParams, gaugeLinks, fields):
    """Symmetric covariant derivative on fields of shape (N_t, N_x, N_vec)."""
    U  = gaugeLinks[:, :, 1].T[:, :, None]                    # (N_t, N_x, 1)
    Um = np.roll(np.conj(U), 1, axis=1)                       # U*_{x-1}
    return (U * np.roll(fields, -1, axis=1)
            - Um * np.roll(fields, 1, axis=1)) / (2 * modelSettings.a)


def buildLaplacian(modelSettings: LatticeParams, gaugeLinks, nt):
    """
    Creates the gauge-covariant laplacian at time slice nt (no spin index)
    """

    #dirac dimensions

    shift_x_1Dpos = np.roll(np.eye(modelSettings.dimx), -1, axis=0) # This is \delta_{x_n+1, x_m}
    shift_x_1Dneg = np.roll(np.eye(modelSettings.dimx), +1, axis=0) # This is \delta_{x_n+1, x_m}

    #flattened gaugelinks: [:,nt, 1] are spatial links at timeslice t
    spaceLinks = sparse.diags_array(gaugeLinks[:,nt,1].flatten())

    #H matrix for smearing
    H = spaceLinks@shift_x_1Dpos + shift_x_1Dneg@np.conj(spaceLinks)
    #subtract off diagonal
    H-= 2*sparse.eye_array(modelSettings.dimx)

    return H

def buildNumberDensOp(modelSettings: LatticeParams, gaugeLinks, chemicalPot=0):
    #dirac dimensions
    dimD = 2
    eyeD = np.eye(dimD)

    shift_t_1Dpos = np.roll(np.eye(modelSettings.dimt), -1, axis=0)
    shift_t_1Dneg = np.roll(np.eye(modelSettings.dimt), +1, axis=0)
    space_identity = sparse.eye_array(modelSettings.dimx)

    #anti-periodic boundary conditions for fermions in time
    shift_t_1Dpos[modelSettings.dimt - 1, 0] = -1.0
    shift_t_1Dneg[0, modelSettings.dimt - 1] = -1.0

    #space-time shift operators

    T_t_pos = sparse.kron(space_identity, shift_t_1Dpos)
    T_t_neg = sparse.kron(space_identity, shift_t_1Dneg)

    #flattened gaugelinks
    timeLinks = sparse.diags_array(gaugeLinks[:,:,0].flatten())

    nOp=-1/(2) * sparse.kron(timeLinks@T_t_pos, eyeD-modelSettings.gammat)*np.exp(modelSettings.a*chemicalPot)
    #negative shifts
    nOp+=1/(2) * sparse.kron(T_t_neg@np.conj(timeLinks),eyeD+modelSettings.gammat)*np.exp(-modelSettings.a*chemicalPot)

    return nOp

def jacobiSmearingH(modelSettings: LatticeParams, gaugeLinks):
    #dirac dimensions
    dimD = 2
    eyeD = np.eye(dimD)

    shift_x_1Dpos = np.roll(np.eye(modelSettings.dimx), -1, axis=0) # This is \delta_{x_n+1, x_m}
    shift_x_1Dneg = np.roll(np.eye(modelSettings.dimx), +1, axis=0) # This is \delta_{x_n+1, x_m}
    time_identity = np.eye(modelSettings.dimt)                      # This is \delta_{t_n, t_m}

    #space-time shift operators
    T_x_pos = sparse.kron(shift_x_1Dpos, time_identity)
    T_x_neg = sparse.kron(shift_x_1Dneg, time_identity)

    #flattened gaugelinks
    spaceLinks = np.diag(gaugeLinks[:,:,1].flatten())

    #H matrix for smearing
    H = sparse.kron(spaceLinks@T_x_pos, eyeD) + sparse.kron(T_x_neg@np.conj(spaceLinks),eyeD)

    return H


def applyJacobi(H, kappa, smearN, v):
    """Apply S = sum_{n=0}^{smearN} kappa^n H^n to v, shape (N,) or (N,k)."""
    out = np.array(v, dtype=complex)
    Hn_v = np.array(v, dtype=complex)
    for n in range(1, smearN + 1):
        Hn_v = H @ Hn_v
        out += kappa**n * Hn_v
    return out

def jacobiSmearingOp(modelSettings: LatticeParams, gaugeLinks, kappa = .1, smearingSteps=1):

    jacobiH = jacobiSmearingH(modelSettings, gaugeLinks).tocsc()

    N = jacobiH.shape[0]
    jacobiM = np.identity(N, dtype=np.complex128)

    if(smearingSteps>0):
        Hn = jacobiH.toarray()
        for n in range(1, smearingSteps+1):
            jacobiM += kappa**n * Hn
            if n < smearingSteps:
                Hn = Hn @ jacobiH

    return jacobiM

def smearedPropagator(modelSettings: LatticeParams, gaugeLinks, kappa=.1, smearingSteps=1, chemicalPot=0):
    Dee = buildDiracOp(modelSettings, gaugeLinks, chemicalPot)

    fullProp = np.linalg.inv(Dee.toarray())
    
    jacobiM = jacobiSmearingOp(modelSettings, gaugeLinks, kappa, smearingSteps)

    smearedProp = jacobiM@fullProp@jacobiM

    return smearedProp