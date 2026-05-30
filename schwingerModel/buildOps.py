import numpy as np
import scipy.sparse as sparse
from scipy.stats import bootstrap
from scipy.optimize import curve_fit
from tqdm import tqdm

from .schwingerModel import schwingerModel

#builds the dirac operator using the global gaugeLinks configuration
# matrix is a square matrix with dimensional ordering (space, time, spin) 
def buildDiracOp(modelObj: schwingerModel, gaugeLinks, chemicalPot=0):
    #dirac dimensions
    dimD = 2
    eyeD = np.eye(dimD)

    shift_x_1Dpos = np.roll(np.eye(modelObj.dimx), -1, axis=0) # This is \delta_{x_n+1, x_m}
    shift_t_1Dpos = np.roll(np.eye(modelObj.dimt), -1, axis=0)
    shift_x_1Dneg = np.roll(np.eye(modelObj.dimx), +1, axis=0) # This is \delta_{x_n+1, x_m}
    shift_t_1Dneg = np.roll(np.eye(modelObj.dimt), +1, axis=0)
    time_identity = np.eye(modelObj.dimt)                      # This is \delta_{t_n, t_m}
    space_identity = np.eye(modelObj.dimx)

    #anti-periodic boundary conditions for fermions in time
    shift_t_1Dpos[modelObj.dimt - 1, 0] = -1.0
    shift_t_1Dneg[0, modelObj.dimt - 1] = -1.0

    #space-time shift operators
    T_x_pos = sparse.kron(shift_x_1Dpos, time_identity)
    T_x_neg = sparse.kron(shift_x_1Dneg, time_identity)
    T_t_pos = sparse.kron(space_identity, shift_t_1Dpos)
    T_t_neg = sparse.kron(space_identity, shift_t_1Dneg)

    #flattened gaugelinks
    spaceLinks = np.diag(gaugeLinks[:,:,1].flatten())
    timeLinks = np.diag(gaugeLinks[:,:,0].flatten())

    #start building dirac operator matrix
    Dee = (modelObj.fMass+2/modelObj.a)*sparse.kron(space_identity, sparse.kron(time_identity,eyeD))
    #positive shifts
    Dee-=1/(2*modelObj.a) * sparse.kron(spaceLinks@T_x_pos, eyeD-modelObj.gammax)
    Dee-=1/(2*modelObj.a) * sparse.kron(timeLinks@T_t_pos, eyeD-modelObj.gammat)*np.exp(modelObj.a*chemicalPot)
    #negative shifts
    Dee-=1/(2*modelObj.a) * sparse.kron(T_x_neg@np.conj(spaceLinks),eyeD+modelObj.gammax)
    Dee-=1/(2*modelObj.a) * sparse.kron(T_t_neg@np.conj(timeLinks),eyeD+modelObj.gammat)*np.exp(-modelObj.a*chemicalPot)

    return Dee

def buildNumberDensOp(modelObj: schwingerModel, gaugeLinks, chemicalPot=0):
    #dirac dimensions
    dimD = 2
    eyeD = np.eye(dimD)

    shift_t_1Dpos = np.roll(np.eye(modelObj.dimt), -1, axis=0)
    shift_t_1Dneg = np.roll(np.eye(modelObj.dimt), +1, axis=0)
    space_identity = np.eye(modelObj.dimx)

    #anti-periodic boundary conditions for fermions in time
    shift_t_1Dpos[modelObj.dimt - 1, 0] = -1.0
    shift_t_1Dneg[0, modelObj.dimt - 1] = -1.0

    #space-time shift operators

    T_t_pos = sparse.kron(space_identity, shift_t_1Dpos)
    T_t_neg = sparse.kron(space_identity, shift_t_1Dneg)

    #flattened gaugelinks
    timeLinks = np.diag(gaugeLinks[:,:,0].flatten())

    nOp=-1/(2) * sparse.kron(timeLinks@T_t_pos, eyeD-modelObj.gammat)*np.exp(modelObj.a*chemicalPot)
    #negative shifts
    nOp+=1/(2) * sparse.kron(T_t_neg@np.conj(timeLinks),eyeD+modelObj.gammat)*np.exp(-modelObj.a*chemicalPot)

    return nOp

def jacobiSmearingH(modelObj: schwingerModel, gaugeLinks):
    #dirac dimensions
    dimD = 2
    eyeD = np.eye(dimD)

    shift_x_1Dpos = np.roll(np.eye(modelObj.dimx), -1, axis=0) # This is \delta_{x_n+1, x_m}
    shift_x_1Dneg = np.roll(np.eye(modelObj.dimx), +1, axis=0) # This is \delta_{x_n+1, x_m}
    time_identity = np.eye(modelObj.dimt)                      # This is \delta_{t_n, t_m}

    #space-time shift operators
    T_x_pos = sparse.kron(shift_x_1Dpos, time_identity)
    T_x_neg = sparse.kron(shift_x_1Dneg, time_identity)

    #flattened gaugelinks
    spaceLinks = np.diag(gaugeLinks[:,:,1].flatten())

    #H matrix for smearing
    H = sparse.kron(spaceLinks@T_x_pos, eyeD) + sparse.kron(T_x_neg@np.conj(spaceLinks),eyeD)

    return H

def jacobiSmearingOp(modelObj: schwingerModel, gaugeLinks, kappa = .1, smearingSteps=1):

    jacobiH = jacobiSmearingH(modelObj, gaugeLinks)

    jacobiM = np.identity(jacobiH.shape[0], dtype=np.complex128)

    if(smearingSteps>0):
        for n in range(1, smearingSteps+1):
            jacobiM += kappa**n * np.linalg.matrix_power(jacobiH.toarray(),n)
    
    return jacobiM

def smearedPropagator(modelObj: schwingerModel, gaugeLinks, kappa=.1, smearingSteps=1, chemicalPot=0):
    Dee = buildDiracOp(modelObj, gaugeLinks, chemicalPot)

    fullProp = np.linalg.inv(Dee.toarray())
    
    jacobiM = jacobiSmearingOp(modelObj, gaugeLinks, kappa, smearingSteps)

    smearedProp = jacobiM@fullProp@jacobiM

    return smearedProp