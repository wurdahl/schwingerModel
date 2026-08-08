from __future__ import annotations

import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import splu, cg
from scipy.stats import bootstrap
from scipy.optimize import curve_fit
from tqdm import tqdm
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

from . import buildOps as ops
from . import topology as top
from .params import LatticeParams, dwfParams

FILE_VERSION = 2   # v2: no links, no gamma attrs, per-config Q

GAMMAS = {"g5":np.array([[1j,0],[0,-1j]]),"gx":np.array([[0,1],[1,0]]),
          "gt":np.array([[0,-1j],[1j,0]]), "id":np.eye(2)}

def _paramsFromAttrs(attrs):
    """LatticeParams or dwfParams from h5 attrs, keyed on the fermionAction attr
    (absent in files that predate it, which are all wilson)."""
    if str(attrs.get("fermionAction", "wilson")) == "dwf":
        return dwfParams(dimx=int(attrs["dimx"]), dimt=int(attrs["dimt"]),
                         dim5=int(attrs["dim5"]),
                         beta=float(attrs["beta"]), fMass=float(attrs["fMass"]),
                         M5=float(attrs["M5"]), a=float(attrs["a"]))
    return LatticeParams(dimx=int(attrs["dimx"]), dimt=int(attrs["dimt"]),
                         beta=float(attrs["beta"]), fMass=float(attrs["fMass"]),
                         a=float(attrs["a"]))

def findPartialEigenBasis(modelSettings: LatticeParams, gaugeLinks, numVecs = 4):

    eigenBases = []

    for nt in range(modelSettings.dimt):
        lap = -ops.buildLaplacian(modelSettings, gaugeLinks, nt=nt)

        #This should find the smallest eigenvalues/eigenvectors of the laplacian
        eigs, eigVecs = sparse.linalg.eigsh(lap, k=numVecs,sigma=0, which='LM')

        eigenBases.append(eigVecs)

    return np.array(eigenBases) #shape: (dimt, dimx, numVecs)

def buildPerambulator(modelSettings: LatticeParams, gaugeLinks, eigVecs, chemicalPot=0):
    """
    Computes the distillation perambulator for a single gauge configuration.

    Returns tau of shape (dimt, dimt, numVecs, 2, numVecs, 2)
      tau[t_sink, t_src, l_sink, s_sink, k_src, s_src]
        = sum_x V(t_sink)[x,l]* M^{-1}[x,t_sink,s_sink; x',t_src,s_src] V(t_src)[x',k]
    Spin is kept as separate indices; .reshape(T, T, 2N, 2N) recovers the
    compound (vec-major, spin-minor) layout l*2+s.
    """
    # eigVecs shape: (dimt, dimx, numVecs)

    N_t, N_x, N_vec = eigVecs.shape

    lu = splu(ops.buildDiracOp(modelSettings, gaugeLinks, chemicalPot).tocsc())

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

def buildDwfPerambulator(modelSettings: dwfParams, gaugeLinks, eigVecs, cgRtol=1e-10):
    """
    Domain wall version of buildPerambulator: same tau shape and index meaning,
    but the physical 2D propagator is the 5D inverse taken between the walls,
      q(x)    = P_- psi(x, s=0) + P_+ psi(x, s=N5-1)
      qbar(y) = psibar(y, N5-1) P_- + psibar(y, 0) P_+
    With gamma5 = diag(1,-1) the projectors are diagonal in spin, so applying
    them is just routing spin components to 5th-dim walls:
      source: spin 0 (P+) -> wall 0        spin 1 (P-) -> wall N5-1
      sink:   spin 0 (P+) <- wall N5-1     spin 1 (P-) <- wall 0
    The wall-to-wall propagator differs from the effective-operator inverse by
    a contact term only, which cannot contribute at t_sink != t_src.
    """
    # eigVecs shape: (dimt, dimx, numVecs)

    N_t, N_x, N_vec = eigVecs.shape
    N5 = modelSettings.dim5
    N2 = N_x*N_t*2      # each 5th-dim slice has the full 2D (x, t, spin) layout

    #solve through the normal equations, x = Ddag (D Ddag)^{-1} b = D^{-1} b:
    #D Ddag is Hermitian positive-definite so CG applies. A direct splu of the
    #5D operator becomes prohibitively slow/memory-hungry above ~32x32, while
    #CG at rtol 1e-10 matches it to solver precision, far below statistical
    #errors (verified against the splu result on a small config).
    D = ops.buildDwfOp(modelSettings, gaugeLinks)
    Ddag = D.conj().T.tocsr()
    DDdag = (D @ Ddag).tocsr()

    def solve(B):
        X = np.empty_like(B)
        for j in range(B.shape[1]):
            y, exitcode = cg(DDdag, B[:, j], rtol=cgRtol)
            if exitcode != 0:
                raise RuntimeError(f"perambulator CG failed to converge! Exit code: {exitcode}")
            X[:, j] = Ddag @ y
        return X

    tau = np.zeros((N_t, N_t, N_vec, 2, N_vec, 2), dtype=complex)

    for t_src in range(N_t):
        # Build sources: one column per (k, s), localized at t_src on the source wall
        B = np.zeros((N5*N2, N_vec*2), dtype=complex)
        for s in range(2):
            rows2d = np.arange(N_x)*N_t*2 + t_src*2 + s
            wall = 0 if s == 0 else N5-1
            B[np.ix_(wall*N2 + rows2d, np.arange(N_vec)*2 + s)] = eigVecs[t_src]  # (N_x, N_vec)

        Phi5 = solve(B).reshape(N5, N_x, N_t, 2, N_vec, 2)
        # (s5, x, t_sink, s_sink, k_src, s_src) -> extract at the sink walls
        Phi = np.empty((N_x, N_t, 2, N_vec, 2), dtype=complex)
        Phi[:, :, 0] = Phi5[N5-1, :, :, 0]
        Phi[:, :, 1] = Phi5[0, :, :, 1]

        # einsum: t=t_sink, a=x (contracted), i=l_sink, j=s_sink, k=k_src, d=s_src
        tau[:, t_src] = np.einsum('tai, atjkd -> tijkd', eigVecs.conj(), Phi, optimize=True)

    return tau

def buildElementalSpatial(modelSettings: LatticeParams, gaugeLinks, eigVecs, DNum=0, momk=0):
    """
    Spatial part of the meson elemental (no spin): V^dag(t) e^{-ikx} D^n V(t),
    shape (N_t, N_vec, N_vec). Gamma matrices are applied at contraction time;
    the barred (source) version is the per-slice conjugate transpose.
    Only DNum > 0 actually reads the links, via applyCovDerivative.
    """
    W = eigVecs                                               # (N_t, N_x, N_vec)
    for _ in range(DNum):
        W = ops.applyCovDerivative(modelSettings, gaugeLinks, W)

    momPhase = np.exp(-1j*2*np.pi*momk*np.arange(modelSettings.dimx)/modelSettings.dimx)

    return np.einsum('txl,x,txk->tlk', eigVecs.conj(), momPhase, W)

def buildElemental(modelSettings: LatticeParams, gaugeLinks, eigVecs, DNum=0,
                   Gamma=np.array([[1j,0],[0,-1j]]), momk=0, bar=False):
    """Full (vec ⊗ spin) elemental in kron form — kept as the independent oracle path."""
    spatial = buildElementalSpatial(modelSettings, gaugeLinks, eigVecs, DNum=DNum, momk=momk)

    if bar:
        gammaBar = modelSettings.gammat @ Gamma.conj().T @ modelSettings.gammat
        return np.kron(spatial.conj().transpose(0, 2, 1), gammaBar)

    return np.kron(spatial, Gamma)


def _measureConfig(modelSettings: LatticeParams, gaugeLinks, numVecs: int, op: MesonOp,
                   chemicalPot, disc):
    """Per-config measurement: one workspace, connected 2pt (+ loops if disc)."""
    ws = DistillWorkspace(modelSettings, gaugeLinks, numVecs, chemicalPot=chemicalPot)
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
    """
    Per-config store. Built from links (generation): eigVecs eagerly, tau and
    elementals computed on demand and cached. Built by `load` (measurement):
    gaugeLinks is None and everything must already be in the file — nothing is
    recomputed, so an elemental that was not generated is an error, not a
    silent (and now impossible) rebuild.
    """
    def __init__(self, modelSettings, gaugeLinks, numVecs, chemicalPot=0):
        self.modelSettings, self.gaugeLinks = modelSettings, gaugeLinks
        self.chemicalPot = chemicalPot
        self.eigVecs = findPartialEigenBasis(modelSettings, gaugeLinks, numVecs)
        self._tau, self._elem = None, {}

    @property
    def tau(self):
        if self._tau is None:
            #the params type carries the fermion action: dwfParams -> wall propagator
            if isinstance(self.modelSettings, dwfParams):
                self._tau = buildDwfPerambulator(self.modelSettings, self.gaugeLinks,
                                                 self.eigVecs)
            else:
                self._tau = buildPerambulator(self.modelSettings, self.gaugeLinks,
                                              self.eigVecs, chemicalPot=self.chemicalPot)
        return self._tau

    def elemental(self, op: MesonOp, bar=False):
        key = (op.momk, op.DNum)              # spatial part doesn't depend on gamma
        if key not in self._elem:
            if self.gaugeLinks is None:
                raise KeyError(f"{op} was not generated in this file (stored: "
                               f"{sorted(self._elem)}); regenerate with momks/DNums covering it")
            S = buildElementalSpatial(self.modelSettings, self.gaugeLinks, self.eigVecs,
                                      DNum=op.DNum, momk=op.momk)
            if np.abs(S).max() < 1e-10:
                raise ValueError(f"{op} unsupported by this basis (momentum window)")
            self._elem[key] = S
        S = self._elem[key]
        return S.conj().transpose(0, 2, 1) if bar else S   # bar = per-slice dagger

    def gamma(self, op: MesonOp, bar=False):
        g = GAMMAS[op.gamma]
        if bar:
            gt = self.modelSettings.gammat
            return gt @ g.conj().T @ gt
        return g

    @classmethod
    def load(cls, filePath, configIndex):
        """
        Rebuild a workspace from a generateDistillFile HDF5 cache. Everything is read
        eagerly and the file closed before returning. The file attrs carry the full
        LatticeParams. Links are never read (v2 does not store them), so the result
        is a pure cache: whatever was generated is available, nothing else.
        Reads v1 files too — their links dataset is simply ignored.
        """
        with h5py.File(filePath, "r") as f:
            gname = f"cfg{configIndex:05d}"
            if gname not in f:
                raise KeyError(f"{filePath} has no group {gname}")
            g = f[gname]

            modelSettings = _paramsFromAttrs(f.attrs)

            ws = cls.__new__(cls)
            ws.modelSettings, ws.gaugeLinks, ws.chemicalPot = modelSettings, None, 0
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

def _generateConfig(modelSettings, gaugeLinks, i, numVecs, momks, DNums):
    ws = DistillWorkspace(modelSettings, gaugeLinks, numVecs)
    data = {"eigVecs": ws.eigVecs}
    data[f"peram"] = ws.tau
    for k in momks:
        for n in DNums:
            data[f"elem/p{k}_d{n}"] = ws.elemental(MesonOp("g5", n, k))  # gamma irrelevant, spatial stored
    #Q is the one thing reweighting needs from the links, so it is stored instead of them
    return i, data, top.getTopoQ(gaugeLinks)


def generateDistillFile(ensemblePath, filePath, numVecs,
                        burnIn=0, autocorrSkip=1, momks=(0,), DNums=(0,), n_jobs=-1):
    """
    Generation stage: read a gauge ensemble written by experiment.saveEnsemble and
    compute eigVecs, perambulator, spatial elementals and the topological charge for
    every selected config, storing them in one HDF5 file (single writer; workers
    only compute).

    The LatticeParams come from the ensemble file's own attrs, so the settings can
    never disagree with the links they describe. Configs are read one at a time as
    they are dispatched — the ensemble is never held in memory whole.

    v2 files do NOT store the gauge links: everything a measurement needs is
    precomputed here, so the momks/DNums given must cover every operator you
    intend to measure — there is no rebuild path afterwards. Q is stored per
    config so theta-reweighting works without the ensemble.

    Reruns are incremental: existing config groups are skipped, so you can extend the
    ensemble coverage — but NOT add datasets to existing groups.
    """
    with h5py.File(ensemblePath, "r") as ens:
        #the ensemble's fermionAction attr decides which params (and perambulator) to use
        modelSettings = _paramsFromAttrs(ens.attrs)
        links = ens["links"]
        indices = [int(i) for i in np.arange(burnIn, links.shape[0], autocorrSkip)]

        meta = {"dimx": modelSettings.dimx, "dimt": modelSettings.dimt, "a": modelSettings.a,
                "fMass": modelSettings.fMass, "beta": modelSettings.beta,
                "numVecs": numVecs, "version": FILE_VERSION,
                "fermionAction": "dwf" if isinstance(modelSettings, dwfParams) else "wilson"}
        if isinstance(modelSettings, dwfParams):
            meta["dim5"] = modelSettings.dim5
            meta["M5"] = modelSettings.M5

        with h5py.File(filePath, "a") as f:
            for key, val in meta.items():
                if key in f.attrs:
                    if not np.all(f.attrs[key] == val):
                        raise ValueError(f"{filePath} was generated with {key}={f.attrs[key]}, "
                                         f"requested {key}={val}; use a different file")
                else:
                    f.attrs[key] = val
            #provenance, outside the consistency check so a moved ensemble is not an error
            if "sourceEnsemble" not in f.attrs:
                f.attrs["sourceEnsemble"] = str(ensemblePath)

            todo = [i for i in indices if f"cfg{i:05d}" not in f]
            if not todo:
                return filePath

            gen = Parallel(n_jobs=n_jobs, return_as="generator")(
                delayed(_generateConfig)(modelSettings, links[i], i, numVecs, momks, DNums)
                for i in todo)
            for i, data, Q in tqdm(gen, total=len(todo), desc="Generating distill data"):
                grp = f.create_group(f"cfg{i:05d}")
                grp.attrs["Q"] = Q
                for key, arr in data.items():
                    grp.create_dataset(key, data=arr)

    return filePath


def readDistillMeta(filePath):
    """
    File-level metadata and inventory of a generateDistillFile cache, so notebooks
    never need the gauge ensemble file. Returns a SimpleNamespace with the stored
    attrs (dimx, dimt, a, fMass, beta, numVecs, version) plus:
      configIndices : sorted list of stored config indices
      elemKeys      : sorted list of stored (momk, DNum) elemental keys
      modelSettings : LatticeParams rebuilt from the attrs
      Q             : (nCfg,) topological charge, ordered like configIndices
                      (None for v1 files, which predate it)
    """
    with h5py.File(filePath, "r") as f:
        meta = SimpleNamespace(**{k: f.attrs[k] for k in f.attrs})
        meta.dimx, meta.dimt = int(meta.dimx), int(meta.dimt)
        meta.numVecs = int(meta.numVecs)
        meta.version = int(meta.version)
        #one early file predates the beta attr, so params can be incomplete
        if all(k in f.attrs for k in ("dimx", "dimt", "beta", "fMass", "a")):
            meta.modelSettings = _paramsFromAttrs(f.attrs)
        else:
            meta.modelSettings = None
        meta.configIndices = sorted(int(name[3:]) for name in f if name.startswith("cfg"))
        first = f[f"cfg{meta.configIndices[0]:05d}"]
        meta.elemKeys = sorted(_parseElemKey(k) for k in first["elem"])
        if "Q" in first.attrs:
            meta.Q = np.array([f[f"cfg{i:05d}"].attrs["Q"] for i in meta.configIndices])
        else:
            meta.Q = None
    return meta
