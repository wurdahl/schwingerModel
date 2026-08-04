"""
Reweighting factors for sign-problem observables.

Config-level quantities (functions of the gauge links only), placed low in the
dependency ladder so both the data layer (distillation) and the statistics
layers (analysis, GEVP) can import them without cycles:

    buildOps -> reweighting -> distillation -> evaluator -> GEVP / analysis
"""
from __future__ import annotations

import numpy as np

from .params import LatticeParams
from . import buildOps as ops
from . import topology as top


def getWeightingFactors(modelSettings: LatticeParams, gaugeConfigs, chemicalPot=1, burnIn=1, autocorrSkip=10):
    """det-ratio reweighting from mu=0 to chemicalPot, squared for two degenerate flavors."""
    if(chemicalPot==0):
        return np.ones(len(np.arange(burnIn,len(gaugeConfigs),autocorrSkip)))

    weights = []

    for i in range(burnIn,len(gaugeConfigs),autocorrSkip):
        currLinks = gaugeConfigs[i]
        dOp = ops.buildDiracOp(modelSettings, currLinks).toarray()
        dOpmu = ops.buildDiracOp(modelSettings, currLinks, chemicalPot).toarray()

        sign_0, logdet_0 = np.linalg.slogdet(dOp)
        sign_mu, logdet_mu = np.linalg.slogdet(dOpmu)
        weights.append((sign_mu / sign_0) * np.exp(logdet_mu - logdet_0))

    #need to square the final weights because there are two degenerate fermions in the problem.
    return np.array(weights)**2


def getWeightingFactorsTheta(Qs, theta=0, burnIn=1, autocorrSkip=10):
    """
    exp(i theta Q) reweighting from the theta=0 ensemble.

    Takes the per-config topological charges directly — distillation v2 files
    store them (readDistillMeta().Q), so this needs no gauge links. For a raw
    ensemble use topology.getAllTopoQs(gaugeConfigs) to produce them.
    """
    Qs = np.asarray(Qs)[burnIn::autocorrSkip]

    if(theta == 0):
        return np.ones(len(Qs))

    return np.exp(1j*theta*Qs)


