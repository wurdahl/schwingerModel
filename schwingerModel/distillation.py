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
from types import SimpleNamespace
import h5py

from .interpolator import MesonOp

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
from .reweighting import getWeightingFactorsTheta

GAMMAS = {"g5":np.array([[1j,0],[0,-1j]]),"gx":np.array([[0,1],[1,0]]),
          "gt":np.array([[0,-1j],[1j,0]]), "id":np.eye(2)}

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


def _measureConfig(modelObj: schwingerModel, configIndex: int, numVecs: int, op: MesonOp,
                   chemicalPot, disc):
    """Per-config measurement: one workspace, connected 2pt (+ loops if disc)."""
    ws = DistillWorkspace(modelObj, configIndex, numVecs, chemicalPot=chemicalPot)
    conn = evalTwoPoint(ws, op, op)
    if not disc:
        return conn, None, None
    # sink loop and barred source loop (identical for g5, momk=0, DNum=0)
    return conn, evalLoop(ws, op), evalLoop(ws, op, bar=True)


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
            "fMass": modelObj.fMass, "beta": modelObj.beta,
            "numVecs": numVecs, "version": 1}

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
