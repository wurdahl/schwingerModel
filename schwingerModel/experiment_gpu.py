import os
import numpy as np

from .params import dwfParams
from . import hmc_gpu
from .experiment import saveEnsemble


def _checkDwfOnly(fermionAction, tunneling):
    """The gpu path implements only the dwf action; fail loudly on cpu-only options."""
    if fermionAction != "dwf":
        raise ValueError(f"the gpu backend only implements fermionAction='dwf', got {fermionAction!r}")
    if tunneling:
        raise ValueError("tunneling steps are not implemented on the gpu backend")


def acceptanceFractions(beta=10, fMass=1, aSpacing=1, Nx=4, Nt=4, metroSteps=100, numSubSteps=10,
                        tunneling=False, cgRtolForce=1e-5, cgRtolAction=1e-10,
                        fermionAction="dwf", dim5=None, M5=None,
                        seed=0, tqdmPosition=0, measureFrom=0.5, chains=8):
    """GPU mirror of experiment.acceptanceFractions, call-compatible so run_sim can
    route to either. Returns (hmcFraction, tunnelFraction); tunnelFraction is always 0.

    Instead of one pilot chain it runs `chains` short chains in lockstep (nearly free
    on the device) and averages the acceptance over the second half of all of them,
    so the estimate is less noisy than the cpu pilot at the same wall-clock cost.
    """
    _checkDwfOnly(fermionAction, tunneling)

    settings = dwfParams(beta=beta, dimt=Nt, dimx=Nx, dim5=dim5, fMass=fMass, M5=M5, a=aSpacing)

    _, acceptHistory = hmc_gpu.hmcChainBatch(settings, chains, metroSteps=metroSteps,
                                             numSubSteps=numSubSteps,
                                             cgRtolForce=cgRtolForce, cgRtolAction=cgRtolAction,
                                             seed=seed, tqdmPosition=tqdmPosition)

    start = int(metroSteps * measureFrom)
    return acceptHistory[start:].mean(), 0.0


def runExperiment(path, beta=10, fMass=1, aSpacing=1, Nx=4, Nt=4, metroSteps=100, numSubSteps=10,
                  chains=16, perChainBurnIn=0,
                  cgRtolForce=1e-5, cgRtolAction=1e-10,
                  dim5=None, M5=None,
                  randSeed=0, overwrite=False,
                  tunneling=False, cores=None, fermionAction="dwf"):
    """GPU mirror of experiment.runExperiment for the dwf action: the chains run in
    lockstep on the device through hmc_gpu.hmcChainBatch instead of joblib workers.

    Writes the same v2 hdf5 layout through saveEnsemble, so loadEnsemble and the
    downstream pipeline read GPU ensembles unchanged. Differences from the CPU
    runner: dwf only (no tunneling steps; `cores` is accepted for call symmetry but
    ignored), and the ensemble is reproducible from (randSeed, chains) through the
    jax key tree rather than per-chain numpy seeds, so the seeds attr records just
    randSeed.
    """
    #fail before spending the chains, not after
    if os.path.exists(path) and not overwrite:
        raise FileExistsError(f"{path} already exists; pass overwrite=True to replace it")

    _checkDwfOnly(fermionAction, tunneling)
    if dim5 is None or M5 is None:
        raise ValueError("the gpu runner is dwf-only and requires dim5 and M5")

    settings = dwfParams(beta=beta, dimt=Nt, dimx=Nx, dim5=dim5, fMass=fMass, M5=M5, a=aSpacing)

    #metroSteps is the TOTAL config count; ceil can overshoot, trimmed after the merge
    stepsPerChain = -(-metroSteps // chains)

    linkHistory, acceptHistory = hmc_gpu.hmcChainBatch(settings, chains,
                                                       metroSteps=perChainBurnIn + stepsPerChain,
                                                       numSubSteps=numSubSteps,
                                                       cgRtolForce=cgRtolForce,
                                                       cgRtolAction=cgRtolAction,
                                                       seed=randSeed)

    #(steps, chains, ...) -> (chains, steps, ...), then the same per-chain
    #burn-in trim and chain merge as the CPU runner
    linkHistory   = np.concatenate(linkHistory.transpose(1, 0, 2, 3, 4)[:, perChainBurnIn:])[:metroSteps]
    acceptHistory = np.concatenate(acceptHistory.T[:, perChainBurnIn:])[:metroSteps]

    saveEnsemble(path, settings, linkHistory, acceptHistory, tunnelAcceptance=None,
                 tunneling=False, cgRtolForce=cgRtolForce, cgRtolAction=cgRtolAction,
                 numSubSteps=numSubSteps, seeds=np.array([randSeed]),
                 fermionAction="dwf", overwrite=overwrite)

    return path
