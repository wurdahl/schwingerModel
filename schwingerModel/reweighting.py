"""
Reweighting factors for sign-problem observables.

Config-level quantities (functions of the gauge links only), placed low in the
dependency ladder so both the data layer (distillation) and the statistics
layers (analysis, GEVP) can import them without cycles:

    buildOps -> reweighting -> distillation -> evaluator -> GEVP / analysis
"""
from __future__ import annotations

import numpy as np

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .schwingerModel import schwingerModel

from . import buildOps as ops


def getWeightingFactors(modelObj: schwingerModel, chemicalPot=1, burnIn=1, autocorrSkip=10):
    """det-ratio reweighting from mu=0 to chemicalPot, squared for two degenerate flavors."""
    if(chemicalPot==0):
        return np.ones(len(np.arange(burnIn,modelObj.metroSteps,autocorrSkip)))

    weights = []

    for i in range(burnIn,modelObj.metroSteps,autocorrSkip):
        currLinks = modelObj.linkHistory[i]
        dOp = ops.buildDiracOp(modelObj, currLinks).toarray()
        dOpmu = ops.buildDiracOp(modelObj, currLinks, chemicalPot).toarray()

        sign_0, logdet_0 = np.linalg.slogdet(dOp)
        sign_mu, logdet_mu = np.linalg.slogdet(dOpmu)
        weights.append((sign_mu / sign_0) * np.exp(logdet_mu - logdet_0))

    #need to square the final weights because there are two degenerate fermions in the problem.
    return np.array(weights)**2


def getWeightingFactorsTheta(modelObj: schwingerModel, theta=0, burnIn=1, autocorrSkip=10):
    """exp(i theta Q) reweighting from the theta=0 ensemble, Q from plaquette angles."""
    if(theta == 0):
        return np.ones(len(np.arange(burnIn,modelObj.metroSteps,autocorrSkip)))

    weights = []

    for i in range(burnIn, modelObj.metroSteps, autocorrSkip):
        currLinks = modelObj.linkHistory[i]

        Ut = currLinks[:,:,0] # Time links (shape: dimx, dimt)
        Ux = currLinks[:,:,1] # Space links (shape: dimx, dimt)

        # Shift arrays to get U_t(x+1, t) and U_x(x, t+1)
        Ut_shifted_x = np.roll(Ut, shift=-1, axis=0)
        Ux_shifted_t = np.roll(Ux, shift=-1, axis=1)

        # Multiply the four sides of the plaquette
        plaq = Ux * Ut_shifted_x * np.conjugate(Ux_shifted_t) * np.conjugate(Ut)

        weights.append(np.sum(np.angle(plaq)))

    return np.exp(1j*theta*np.array(weights)/(2*np.pi))
