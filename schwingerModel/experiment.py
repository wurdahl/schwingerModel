import os
import h5py
import numpy as np
from types import SimpleNamespace
from joblib import Parallel, delayed

from .params import LatticeParams, dwfParams
from . import hmc
from . import hmc_dwf

def saveEnsemble(path, modelSettings, linkHistory,
                  acceptHistory, tunnelAcceptance, tunneling,
                  cgRtolForce, cgRtolAction, numSubSteps, seeds,
                  fermionAction="wilson", dHHistory=None, overwrite=False):

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    with h5py.File(path, "w" if overwrite else "x") as f:
        # scalars/metadata -> attrs
        #_fields covers both param types, so dwf runs also record dim5 and M5
        for k in modelSettings._fields:
            f.attrs[k] = getattr(modelSettings, k)

        f.attrs["fermionAction"] = fermionAction
        f.attrs["tunneling"] = tunneling
        f.attrs["cgRtolForce"] = cgRtolForce
        f.attrs["cgRtolAction"] = cgRtolAction
        f.attrs["numSubSteps"] = numSubSteps

        f.attrs["seeds"] = np.asarray(seeds)      # arrays are fine as attrs if small
        f.attrs["version"] = 2

        # bulk arrays -> datasets
        f.create_dataset("links", data=linkHistory, chunks=(1, modelSettings.dimx, modelSettings.dimt, 2))
        f.create_dataset("acceptHistory", data=acceptHistory)
        if tunnelAcceptance is not None:
            f.create_dataset("tunnelAcceptance", data=tunnelAcceptance)
        if dHHistory is not None:
            f.create_dataset("dHHistory", data=dHHistory)


def loadEnsemble(path):
    """
    Inverse of saveEnsemble: rebuilds the params from the file attrs and reads
    the stored chain. Handles version 1 (wilson-only, single cgRtol) and
    version 2 (fermionAction attr, split force/action tolerances, optional
    dHHistory). Returns a SimpleNamespace with
      modelSettings    : LatticeParams or dwfParams (by fermionAction)
      fermionAction    : "wilson" | "dwf"
      linkHistory      : (nCfg, dimx, dimt, 2) complex
      acceptHistory    : (nCfg,) bool
      tunnelAcceptance : (nCfg,) bool, or None if the run had none
      dHHistory        : (nCfg,) float, or None if the run predates it
    plus the run settings it was generated with (tunneling, cgRtolForce,
    cgRtolAction, numSubSteps, seeds) and the file version.
    """
    with h5py.File(path, "r") as f:
        version = int(f.attrs["version"])
        if version not in (1, 2):
            raise ValueError(f"{path} has version {version}, this reader only knows versions 1 and 2")

        fermionAction = str(f.attrs.get("fermionAction", "wilson"))

        if fermionAction == "dwf":
            modelSettings = dwfParams(dimx=int(f.attrs["dimx"]), dimt=int(f.attrs["dimt"]),
                                      dim5=int(f.attrs["dim5"]),
                                      beta=float(f.attrs["beta"]), fMass=float(f.attrs["fMass"]),
                                      M5=float(f.attrs["M5"]), a=float(f.attrs["a"]))
        else:
            modelSettings = LatticeParams(dimx=int(f.attrs["dimx"]), dimt=int(f.attrs["dimt"]),
                                          beta=float(f.attrs["beta"]), fMass=float(f.attrs["fMass"]),
                                          a=float(f.attrs["a"]))

        if version == 1:
            #v1 ran one tolerance for everything
            cgRtolForce = cgRtolAction = float(f.attrs["cgRtol"])
        else:
            cgRtolForce = float(f.attrs["cgRtolForce"])
            cgRtolAction = float(f.attrs["cgRtolAction"])

        #[:] is what actually pulls the data off disk; without it these stay file handles
        return SimpleNamespace(modelSettings=modelSettings,
                               fermionAction=fermionAction,
                               linkHistory=f["links"][:],
                               acceptHistory=f["acceptHistory"][:],
                               tunnelAcceptance=f["tunnelAcceptance"][:] if "tunnelAcceptance" in f else None,
                               dHHistory=f["dHHistory"][:] if "dHHistory" in f else None,
                               tunneling=bool(f.attrs["tunneling"]),
                               cgRtolForce=cgRtolForce,
                               cgRtolAction=cgRtolAction,
                               numSubSteps=int(f.attrs["numSubSteps"]),
                               seeds=np.asarray(f.attrs["seeds"]),
                               version=version)

def acceptanceFractions(beta=10, fMass=1, aSpacing=1, Nx=4, Nt=4, metroSteps=100, numSubSteps=10,
                        tunneling=False, cgRtolForce=1e-5, cgRtolAction=1e-10,
                        fermionAction="wilson", dim5=None, M5=None,
                        seed=0, tqdmPosition=0, measureFrom=0.5):
    """
    Acceptance probe: runs one short chain and returns (hmcFraction, tunnelFraction)
    without writing anything. Measured over the last (1 - measureFrom) of the chain
    so the cold-start transient does not drag the estimate down.
    tunnelFraction is 0.0 when tunneling is off (always, for dwf).
    """
    if fermionAction == "dwf":
        if tunneling:
            raise ValueError("tunneling steps are not implemented for the dwf action")
        settings = dwfParams(beta=beta, dimt=Nt, dimx=Nx, dim5=dim5, fMass=fMass, M5=M5, a=aSpacing)
        _, _, acceptHistory = hmc_dwf.hmcChain(settings, metroSteps, numSubSteps,
                                               cgRtolForce, cgRtolAction, seed, tqdmPosition)
        tunnelAcceptance = np.zeros(metroSteps, dtype=bool)
    else:
        settings = LatticeParams(beta=beta, dimt=Nt, dimx=Nx, fMass=fMass, a=aSpacing)
        _, _, acceptHistory, tunnelAcceptance = hmc.hmcChain(settings, metroSteps, numSubSteps,
                                                             cgRtolForce, cgRtolAction,
                                                             tunneling, seed, tqdmPosition)

    start = int(metroSteps * measureFrom)
    return acceptHistory[start:].mean(), tunnelAcceptance[start:].mean()


def runExperiment(path, beta=10, fMass=1, aSpacing=1, Nx=4,Nt=4, metroSteps=100, numSubSteps=10, tunneling=True,
                   cores=None, chains=10, perChainBurnIn=0,
                   cgRtolForce=1e-5, cgRtolAction=1e-10,
                   fermionAction="wilson", dim5=None, M5=None,
                   randSeed=0, overwrite=False):
    """
    Runs `chains` independent HMC chains in parallel, drops each chain's burn-in,
    merges the thermalized configs and writes them to `path` with saveEnsemble.

    metroSteps is the TOTAL number of configs wanted across all chains; each chain
    runs ceil(metroSteps/chains) of them, plus perChainBurnIn discarded steps of its
    own. Chain i uses seed randSeed + i, so the ensemble is reproducible from
    (randSeed, chains, settings). Returns the path written.

    fermionAction selects the sea quark discretization: "wilson" (default) or
    "dwf", which requires dim5 and M5 and does not support tunneling steps.
    """

    #fail before spending the chains, not after: saveEnsemble would only catch this at the end
    if os.path.exists(path) and not overwrite:
        raise FileExistsError(f"{path} already exists; pass overwrite=True to replace it")

    if fermionAction not in ("wilson", "dwf"):
        raise ValueError(f"unknown fermionAction {fermionAction!r}; expected 'wilson' or 'dwf'")

    if cores is None:
        cores = min(chains, os.cpu_count() or 1)

    seeds = randSeed + np.arange(chains)
    stepsPerChain = -(-metroSteps // chains)        #ceil division

    #workers only compute; the parent does the single-writer hdf5 write below
    if fermionAction == "dwf":
        if tunneling:
            raise ValueError("tunneling steps are not implemented for the dwf action")
        if dim5 is None or M5 is None:
            raise ValueError("fermionAction='dwf' requires dim5 and M5")
        settings = dwfParams(beta=beta, dimt=Nt, dimx=Nx, dim5=dim5, fMass=fMass, M5=M5, a=aSpacing)

        results = Parallel(n_jobs=cores)(
            delayed(hmc_dwf.hmcChain)(settings, perChainBurnIn + stepsPerChain, numSubSteps,
                                      cgRtolForce, cgRtolAction, seeds[i], i)
            for i in range(chains))

        tunnelAcceptance = None
        #hmc_dwf.hmcChain does not currently record dH; the v2 dataset is optional
        dHHistory = None
    else:
        settings = LatticeParams(beta=beta, dimt=Nt,dimx=Nx, fMass=fMass, a=aSpacing)

        results = Parallel(n_jobs=cores)(
            delayed(hmc.hmcChain)(settings, perChainBurnIn + stepsPerChain, numSubSteps,
                                  cgRtolForce, cgRtolAction, tunneling, seeds[i], i)
            for i in range(chains))

        tunnelAcceptance = np.concatenate([r[3][perChainBurnIn:] for r in results])[:metroSteps]
        dHHistory = None

    #the cold-start transient is per chain, so each one is trimmed before merging;
    #the ceil above can overshoot, so the merged result is cut back to metroSteps
    linkHistory      = np.concatenate([r[1][perChainBurnIn:] for r in results])[:metroSteps]
    acceptHistory    = np.concatenate([r[2][perChainBurnIn:] for r in results])[:metroSteps]

    saveEnsemble(path, settings, linkHistory, acceptHistory, tunnelAcceptance,
                 tunneling, cgRtolForce, cgRtolAction, numSubSteps, seeds,
                 fermionAction=fermionAction, dHHistory=dHHistory, overwrite=overwrite)

    return path
    