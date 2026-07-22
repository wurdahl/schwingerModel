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
from typing import NamedTuple
from types import SimpleNamespace
import h5py

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

from . import buildOps as ops
from .analysis import getWeightingFactorsTheta

GAMMAS = {"g5":np.array([[1j,0],[0,-1j]]),"gx":np.array([[0,1],[1,0]]),
          "gt":np.array([[0,-1j],[1j,0]]), "id":np.eye(2)}

class MesonOp(NamedTuple):
    gamma:str
    DNum: int=0
    momk: int=0

def findPartialEigenBasis(modelObj: schwingerModel, configIndex = 0, numVecs = 4):

    eigenBases = []

    for nt in range(modelObj.dimt):
        lap = -ops.buildLaplacian(modelObj, modelObj.linkHistory[configIndex], nt=nt)

        #This should find the smallest eigenvalues/eigenvectors of the laplacian
        eigs, eigVecs = sparse.linalg.eigsh(lap, k=numVecs,sigma=0, which='LM')

        #momentum projection
        # eigVecs *= np.exp(-1j*2*np.pi*momk*np.arange(modelObj.dimx)/modelObj.dimx)

        eigenBases.append(eigVecs)

    return np.array(eigenBases) #shape: (dimt, dimx, numVecs)

def buildPerambulator(modelObj: schwingerModel, configIndex: int, eigVecs, chemicalPot=0):
    """
    Computes the distillation perambulator for a single gauge configuration.

    Returns tau of shape (dimt, dimt, numVecs, 2, numVecs, 2)
      tau[t_sink, t_src, l_sink, s_sink, k_src, s_src]
        = sum_x V(t_sink)[x,l]* M^{-1}[x,t_sink,s_sink; x',t_src,s_src] V(t_src)[x',k]
    Spin is kept as separate indices; .reshape(T, T, 2N, 2N) recovers the
    compound (vec-major, spin-minor) layout l*2+s.
    """
    gaugeLinks = modelObj.linkHistory[configIndex]

    # eigVecs shape: (dimt, dimx, numVecs)

    N_t, N_x, N_vec = eigVecs.shape

    lu = splu(ops.buildDiracOp(modelObj, gaugeLinks, chemicalPot).tocsc())

    tau = np.zeros((N_t, N_t, N_vec, 2, N_vec, 2), dtype=complex)

    for t_src in range(N_t):
        # Build sources: one column per (k, s), localized at t_src
        B = np.zeros((N_x*N_t*2, N_vec*2), dtype=complex)
        for s in range(2):
            rows = np.arange(N_x)*N_t*2 + t_src*2 + s
            B[np.ix_(rows, np.arange(N_vec)*2 + s)] = eigVecs[t_src]  # (N_x, N_vec)

        Phi = lu.solve(B).reshape(N_x, N_t, 2, N_vec, 2)
        # (x, t_sink, s_sink, k_src, s_src)

        # einsum: t=t_sink, a=x (contracted), i=l_sink, j=s_sink, k=k_src, d=s_src
        tau[:, t_src] = np.einsum('tai, atjkd -> tijkd', eigVecs.conj(), Phi, optimize=True)

    return tau

def buildElementalSpatial(modelObj: schwingerModel, configIndex: int, eigVecs, DNum=0, momk=0):
    """
    Spatial part of the meson elemental (no spin): V^dag(t) e^{-ikx} D^n V(t),
    shape (N_t, N_vec, N_vec). Gamma matrices are applied at contraction time;
    the barred (source) version is the per-slice conjugate transpose.
    """
    W = eigVecs                                               # (N_t, N_x, N_vec)
    for _ in range(DNum):
        W = ops.applyCovDerivative(modelObj, modelObj.linkHistory[configIndex], W)

    momPhase = np.exp(-1j*2*np.pi*momk*np.arange(modelObj.dimx)/modelObj.dimx)

    return np.einsum('txl,x,txk->tlk', eigVecs.conj(), momPhase, W)

def buildElemental(modelObj: schwingerModel, configIndex: int, eigVecs, DNum=0,
                   Gamma=np.array([[1j,0],[0,-1j]]), momk=0, bar=False):
    """Full (vec ⊗ spin) elemental in kron form — kept as the independent oracle path."""
    spatial = buildElementalSpatial(modelObj, configIndex, eigVecs, DNum=DNum, momk=momk)

    if bar:
        gammaBar = modelObj.gammat @ Gamma.conj().T @ modelObj.gammat
        return np.kron(spatial.conj().transpose(0, 2, 1), gammaBar)

    return np.kron(spatial, Gamma)


def getCorrelation(modelObj: schwingerModel, configIndex: int, numVecs: int, chemicalPot=0,
                    gamma=np.array([[1j,0],[0,-1j]]), momk=0, DNum = 0):

    eigVecs = findPartialEigenBasis(modelObj, configIndex, numVecs)

    peramb = buildPerambulator(modelObj, configIndex, eigVecs, chemicalPot=chemicalPot)
    T, _, N = peramb.shape[:3]
    peramb = peramb.reshape(T, T, 2*N, 2*N)   # compound layout for the kron oracle

    Esnk = buildElemental(modelObj, configIndex, eigVecs, DNum=DNum, Gamma=gamma, momk=momk)
    Esrc = buildElemental(modelObj, configIndex, eigVecs, DNum=DNum, Gamma=gamma, momk=momk, bar=True)

    trace = -np.einsum("ijkl,jlm,jimn,ink->ij", peramb, Esrc, peramb, Esnk, optimize=True)

    correlator = np.array([
        np.roll(trace, -dt, axis=0).diagonal().mean()
        for dt in range(modelObj.dimt)
    ])

    return correlator

def getCorrelationLoop(modelObj: schwingerModel, configIndex: int, numVecs: int, chemicalPot=0,
                    gamma=np.array([[1j,0],[0,-1j]]), momk=0,DNum=0):
    #assuming isospin symmetry for everything

    eigVecs = findPartialEigenBasis(modelObj, configIndex, numVecs)
    
    peramb = buildPerambulator(modelObj, configIndex, eigVecs, chemicalPot=chemicalPot)
    T, _, N = peramb.shape[:3]
    peramb = peramb.reshape(T, T, 2*N, 2*N)   # compound layout for the kron oracle

    elemental = buildElemental(modelObj, configIndex, eigVecs, DNum=DNum, Gamma=gamma,momk=momk)

    trace = np.einsum("iikl,ilk->i",peramb,elemental,optimize=True)

    return trace

def _measureConfig(modelObj: schwingerModel, configIndex: int, numVecs: int, op: MesonOp,
                   chemicalPot, disc):
    """Per-config measurement: one workspace, connected 2pt (+ loops if disc)."""
    ws = DistillWorkspace(modelObj, configIndex, numVecs, chemicalPot=chemicalPot)
    conn = evalTwoPoint(ws, op, op)
    if not disc:
        return conn, None, None
    # sink loop and barred source loop (identical for g5, momk=0, DNum=0)
    return conn, evalLoop(ws, op), evalLoop(ws, op, bar=True)

def correlStats(modelObj: schwingerModel, burnIn=1, autocorrSkip=1,
                    op: MesonOp = None, gamma="g5", nVec=2,
                      chemicalPot=0, theta=0,disc=False, discFactor=2,
                      momk=0, DNum=0):

    if op is None:
        op = MesonOp(gamma, DNum, momk)

    weights = getWeightingFactorsTheta(modelObj, theta=theta,burnIn=burnIn, autocorrSkip=autocorrSkip)

    indices = np.arange(burnIn, modelObj.metroSteps, autocorrSkip)
    with tqdm_joblib(tqdm(total=len(indices), desc="Configs")):
        results = Parallel(n_jobs=-1)(delayed(_measureConfig)(modelObj, i, nVec, op, chemicalPot, disc) for i in indices)

    correl = np.array([r[0] for r in results])

    if(disc):
        # loops[n,t] = L_n(t) = Tr[Phi tau(t,t)], loopsBar with the barred elemental
        loops    = np.array([r[1] for r in results])
        loopsBar = np.array([r[2] for r in results])

    totalCorrelMean = np.real(np.average(correl,axis=0,weights=weights))

    if(disc):
        dimt = modelObj.dimt

        # per-config, translation-averaged loop-loop product on the SAME config:
        #   loopCorrel[n, dt] = (1/T) sum_t L_n(t+dt) Lbar_n(t)
        loopCorrel = np.stack([
            np.mean(np.roll(loops, -dt, axis=1) * loopsBar, axis=1)
            for dt in range(dimt)
        ], axis=1)                                       # (n_configs, dimt)

        def _discCorrel(llc, lp, lpb, w):
            # llc, lp, lpb: (..., n_configs, dimt); w: (..., n_configs)
            # returns vacuum-subtracted disconnected correlator (..., dimt)
            ws    = np.sum(w, axis=-1, keepdims=True)
            LL    = np.sum(llc * w[..., None], axis=-2) / ws         # <(1/T) sum_t L(t+dt)Lbar(t)>
            Lbar  = np.sum(lp  * w[..., None], axis=-2) / ws         # <L(t)>     (..., dimt)
            LbarB = np.sum(lpb * w[..., None], axis=-2) / ws         # <Lbar(t)>  (..., dimt)
            vac   = np.stack([                                       # (1/T) sum_t <L(t+dt)><Lbar(t)>
                np.mean(np.roll(Lbar, -dt, axis=-1) * LbarB, axis=-1)
                for dt in range(dimt)
            ], axis=-1)
            return LL - vac

        totalCorrelMean = totalCorrelMean + discFactor*np.real(_discCorrel(loopCorrel, loops, loopsBar, weights))

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
            chunk_means = chunk_means + discFactor*np.real(
                _discCorrel(loopCorrel[idx], loops[idx], loopsBar[idx], w_chunk)
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


def _parseElemKey(name):
    """'p{k}_d{n}' -> (momk, DNum), matching the workspace _elem cache key."""
    p, d = name.split("_")
    return (int(p[1:]), int(d[1:]))


class DistillWorkspace:
    """Per-config store: eigVecs eagerly, tau and elementals lazily, everything cached."""
    def __init__(self, modelObj, configIndex, numVecs, chemicalPot=0):
        self.modelObj, self.configIndex = modelObj, configIndex
        self.chemicalPot = chemicalPot
        self.eigVecs = findPartialEigenBasis(modelObj, configIndex, numVecs)
        self._tau, self._elem = None, {}

    @property
    def tau(self):
        if self._tau is None:
            self._tau = buildPerambulator(self.modelObj, self.configIndex,
                                          self.eigVecs, chemicalPot=self.chemicalPot)
        return self._tau

    def elemental(self, op: MesonOp, bar=False):
        key = (op.momk, op.DNum)              # spatial part doesn't depend on gamma
        if key not in self._elem:
            S = buildElementalSpatial(self.modelObj, self.configIndex, self.eigVecs,
                                      DNum=op.DNum, momk=op.momk)
            if np.abs(S).max() < 1e-10:
                raise ValueError(f"{op} unsupported by this basis (momentum window)")
            self._elem[key] = S
        S = self._elem[key]
        return S.conj().transpose(0, 2, 1) if bar else S   # bar = per-slice dagger

    def gamma(self, op: MesonOp, bar=False):
        g = GAMMAS[op.gamma]
        if bar:
            gt = self.modelObj.gammat
            return gt @ g.conj().T @ gt
        return g

    @classmethod
    def load(cls, filePath, configIndex):
        """
        Rebuild a workspace from a generateDistillFile HDF5 cache. Everything is read
        eagerly and the file closed before returning. The stub model carries enough
        metadata (dims, a, gammas, this config's links) that elementals not in the
        file can still be built lazily against the stored eigenvector basis.
        """
        with h5py.File(filePath, "r") as f:
            gname = f"cfg{configIndex:05d}"
            if gname not in f:
                raise KeyError(f"{filePath} has no group {gname}")
            g = f[gname]

            stub = SimpleNamespace(dimx=int(f.attrs["dimx"]), dimt=int(f.attrs["dimt"]),
                                   a=f.attrs["a"], fMass=f.attrs["fMass"],
                                   gammat=np.asarray(f.attrs["gammat"]),
                                   gammax=np.asarray(f.attrs["gammax"]),
                                   linkHistory={configIndex: g["links"][:]})

            ws = cls.__new__(cls)
            ws.modelObj, ws.configIndex, ws.chemicalPot = stub, configIndex, 0
            ws.eigVecs = g["eigVecs"][:]
            ws._tau = g["peram"][:]
            ws._elem = {_parseElemKey(k): g["elem"][k][:] for k in g["elem"]}
        return ws


def evalTwoPoint(ws: DistillWorkspace, snkOp:MesonOp, srcOp:MesonOp):
    # Tr[ E_snk(i) tau(i,j) Ebar_src(j) tau(j,i) ], spin factored out:
    # tau[i,j,a,s,b,t], spatial (vec,vec), gamma (spin,spin)
    trace = -np.einsum("ijasbt,jbc,tu,jicudv,ida,vs->ij",
                       ws.tau, ws.elemental(srcOp, bar=True), ws.gamma(srcOp, bar=True),
                       ws.tau, ws.elemental(snkOp),           ws.gamma(snkOp),
                       optimize=True)
    T = trace.shape[0]
    return np.array([np.roll(trace, -dt, axis=0).diagonal().mean() for dt in range(T)])

def evalLoop(ws, op, bar=False):
    return np.einsum("iiasbt,iba,ts->i", ws.tau, ws.elemental(op, bar), ws.gamma(op, bar),
                     optimize=True)

def _generateConfig(modelObj, i, numVecs, momks, DNums):
    ws = DistillWorkspace(modelObj, i, numVecs)
    data = {"eigVecs": ws.eigVecs, "links": modelObj.linkHistory[i]}
    data[f"peram"] = ws.tau
    for k in momks:
        for n in DNums:
            data[f"elem/p{k}_d{n}"] = ws.elemental(MesonOp("g5", n, k))  # gamma irrelevant, spatial stored
    return i, data


def generateDistillFile(modelObj: schwingerModel, filePath, numVecs, burnIn=0, autocorrSkip=1,
                        momks=(0,), DNums=(0,), n_jobs=-1):
    """
    Generation stage: compute eigVecs, perambulator and spatial elementals for every
    config and store them in one HDF5 file (single writer; workers only compute).
    Reruns are incremental: existing config groups are skipped, so you can extend the
    ensemble coverage — but NOT add datasets to existing groups (that would need the
    stored eigVecs; use DistillWorkspace.load and its lazy elemental path instead).
    """
    indices = [int(i) for i in np.arange(burnIn, modelObj.metroSteps, autocorrSkip)]

    meta = {"dimx": modelObj.dimx, "dimt": modelObj.dimt, "a": modelObj.a,
            "fMass": modelObj.fMass, "numVecs": numVecs, "version": 1}

    with h5py.File(filePath, "a") as f:
        for key, val in meta.items():
            if key in f.attrs:
                if not np.all(f.attrs[key] == val):
                    raise ValueError(f"{filePath} was generated with {key}={f.attrs[key]}, "
                                     f"requested {key}={val}; use a different file")
            else:
                f.attrs[key] = val
        if "gammat" not in f.attrs:
            f.attrs["gammat"] = np.asarray(modelObj.gammat, dtype=complex)
            f.attrs["gammax"] = np.asarray(modelObj.gammax, dtype=complex)

        todo = [i for i in indices if f"cfg{i:05d}" not in f]
        if not todo:
            return filePath

        gen = Parallel(n_jobs=n_jobs, return_as="generator")(
            delayed(_generateConfig)(modelObj, i, numVecs, momks, DNums)
            for i in todo)
        for i, data in tqdm(gen, total=len(todo), desc="Generating distill data"):
            grp = f.create_group(f"cfg{i:05d}")
            for key, arr in data.items():
                grp.create_dataset(key, data=arr)

    return filePath


def readDistillMeta(filePath):
    """
    File-level metadata and inventory of a generateDistillFile cache, so notebooks
    never need the schwingerModel pickle. Returns a SimpleNamespace with the stored
    attrs (dimx, dimt, a, fMass, numVecs, gammat, gammax, version) plus:
      configIndices : sorted list of stored config indices
      elemKeys      : sorted list of stored (momk, DNum) elemental keys
    """
    with h5py.File(filePath, "r") as f:
        meta = SimpleNamespace(**{k: f.attrs[k] for k in f.attrs})
        meta.dimx, meta.dimt = int(meta.dimx), int(meta.dimt)
        meta.numVecs = int(meta.numVecs)
        meta.configIndices = sorted(int(name[3:]) for name in f if name.startswith("cfg"))
        first = f[f"cfg{meta.configIndices[0]:05d}"]
        meta.elemKeys = sorted(_parseElemKey(k) for k in first["elem"])
    return meta
