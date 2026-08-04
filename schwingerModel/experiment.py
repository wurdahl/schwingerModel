import os
import h5py
import numpy as np
from types import SimpleNamespace
from joblib import Parallel, delayed

from .params import LatticeParams
from . import hmc

def saveEnsemble(path, modelSettings: LatticeParams, linkHistory,
                  acceptHistory, tunnelAcceptance, tunneling, cgRtol, numSubSteps, seeds, overwrite=False):

  
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    with h5py.File(path, "w" if overwrite else "x") as f:
        # scalars/metadata -> attrs
        for k in ("dimx", "dimt", "beta", "fMass", "a"):
            f.attrs[k] = getattr(modelSettings, k)

        f.attrs["tunneling"] = tunneling
        f.attrs["cgRtol"] = cgRtol
        f.attrs["numSubSteps"] = numSubSteps

        f.attrs["seeds"] = np.asarray(seeds)      # arrays are fine as attrs if small
        f.attrs["version"] = 1

        # bulk arrays -> datasets
        f.create_dataset("links", data=linkHistory, chunks=(1, modelSettings.dimx, modelSettings.dimt, 2))
        f.create_dataset("acceptHistory", data=acceptHistory)
        f.create_dataset("tunnelAcceptance", data=tunnelAcceptance)


def loadEnsemble(path):
    """
    Inverse of saveEnsemble: rebuilds the LatticeParams from the file attrs and
    reads the stored chain. Returns a SimpleNamespace with
      modelSettings    : LatticeParams
      linkHistory      : (nCfg, dimx, dimt, 2) complex
      acceptHistory    : (nCfg,) bool
      tunnelAcceptance : (nCfg,) bool
    plus the run settings it was generated with (tunneling, cgRtol, numSubSteps,
    seeds) and the file version.
    """
    with h5py.File(path, "r") as f:
        version = int(f.attrs["version"])
        if version != 1:
            raise ValueError(f"{path} has version {version}, this reader only knows version 1")

        modelSettings = LatticeParams(dimx=int(f.attrs["dimx"]), dimt=int(f.attrs["dimt"]),
                                      beta=float(f.attrs["beta"]), fMass=float(f.attrs["fMass"]),
                                      a=float(f.attrs["a"]))

        #[:] is what actually pulls the data off disk; without it these stay file handles
        return SimpleNamespace(modelSettings=modelSettings,
                               linkHistory=f["links"][:],
                               acceptHistory=f["acceptHistory"][:],
                               tunnelAcceptance=f["tunnelAcceptance"][:],
                               tunneling=bool(f.attrs["tunneling"]),
                               cgRtol=float(f.attrs["cgRtol"]),
                               numSubSteps=int(f.attrs["numSubSteps"]),
                               seeds=np.asarray(f.attrs["seeds"]),
                               version=version)

def acceptanceFractions(beta=10, fMass=1, aSpacing=1, Nx=4, Nt=4, metroSteps=100, numSubSteps=10,
                        tunneling=False, cgRtol=1e-5, seed=0, tqdmPosition=0, measureFrom=0.5):
    """
    Acceptance probe: runs one short chain and returns (hmcFraction, tunnelFraction)
    without writing anything. Measured over the last (1 - measureFrom) of the chain
    so the cold-start transient does not drag the estimate down.
    tunnelFraction is 0.0 when tunneling is off.
    """
    settings = LatticeParams(beta=beta, dimt=Nt, dimx=Nx, fMass=fMass, a=aSpacing)

    _, _, acceptHistory, tunnelAcceptance = hmc.hmcChain(settings, metroSteps, numSubSteps,
                                                         cgRtol, tunneling, seed, tqdmPosition)

    start = int(metroSteps * measureFrom)
    return acceptHistory[start:].mean(), tunnelAcceptance[start:].mean()


def runExperiment(path, beta=10, fMass=1, aSpacing=1, Nx=4,Nt=4, metroSteps=100, numSubSteps=10, tunneling=True,
                   cores=None, chains=10, perChainBurnIn=0,cgRtol=1e-5, randSeed=0, overwrite=False):
    """
    Runs `chains` independent HMC chains in parallel, drops each chain's burn-in,
    merges the thermalized configs and writes them to `path` with saveEnsemble.

    metroSteps is the TOTAL number of configs wanted across all chains; each chain
    runs ceil(metroSteps/chains) of them, plus perChainBurnIn discarded steps of its
    own. Chain i uses seed randSeed + i, so the ensemble is reproducible from
    (randSeed, chains, settings). Returns the path written.
    """

    #fail before spending the chains, not after: saveEnsemble would only catch this at the end
    if os.path.exists(path) and not overwrite:
        raise FileExistsError(f"{path} already exists; pass overwrite=True to replace it")

    if cores is None:
        cores = min(chains, os.cpu_count() or 1)

    settings = LatticeParams(beta=beta, dimt=Nt,dimx=Nx, fMass=fMass, a=aSpacing)

    seeds = randSeed + np.arange(chains)
    stepsPerChain = -(-metroSteps // chains)        #ceil division

    #workers only compute; the parent does the single-writer hdf5 write below
    results = Parallel(n_jobs=cores)(
        delayed(hmc.hmcChain)(settings, perChainBurnIn + stepsPerChain, numSubSteps, cgRtol,
                              tunneling, seeds[i], i)
        for i in range(chains))

    #the cold-start transient is per chain, so each one is trimmed before merging;
    #the ceil above can overshoot, so the merged result is cut back to metroSteps
    linkHistory      = np.concatenate([r[1][perChainBurnIn:] for r in results])[:metroSteps]
    acceptHistory    = np.concatenate([r[2][perChainBurnIn:] for r in results])[:metroSteps]
    tunnelAcceptance = np.concatenate([r[3][perChainBurnIn:] for r in results])[:metroSteps]

    saveEnsemble(path, settings, linkHistory, acceptHistory, tunnelAcceptance,
                 tunneling, cgRtol, numSubSteps, seeds, overwrite=overwrite)

    return path
    