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

from .buildOps import buildDiracOp
from .analysis import getWeightingFactorsTheta
from . import wick

def buildLaplacian(modelObj: schwingerModel, gaugeLinks, nt):
    """
    Creates the gauge-covariant laplacian at time slice nt (no spin index)
    """

    #dirac dimensions

    shift_x_1Dpos = np.roll(np.eye(modelObj.dimx), -1, axis=0) # This is \delta_{x_n+1, x_m}
    shift_x_1Dneg = np.roll(np.eye(modelObj.dimx), +1, axis=0) # This is \delta_{x_n+1, x_m}

    #flattened gaugelinks: [:,nt, 1] are spatial links at timeslice t
    spaceLinks = sparse.diags_array(gaugeLinks[:,nt,1].flatten())

    #H matrix for smearing
    H = spaceLinks@shift_x_1Dpos + shift_x_1Dneg@np.conj(spaceLinks)
    #subtract off diagonal
    H-= 2*sparse.eye_array(modelObj.dimx)

    return H

def findPartialEigenBasis(modelObj: schwingerModel, configIndex = 0, numVecs = 4):

    eigenBases = []

    for nt in range(modelObj.dimt):
        lap = -buildLaplacian(modelObj, modelObj.linkHistory[configIndex], nt=nt)

        #This should find the smallest eigenvalues/eigenvectors of the laplacian
        eigs, eigVecs = sparse.linalg.eigsh(lap, k=numVecs,sigma=0, which='LM')

        #momentum projection
        # eigVecs *= np.exp(-1j*2*np.pi*momk*np.arange(modelObj.dimx)/modelObj.dimx)

        eigenBases.append(eigVecs)

    return np.array(eigenBases) #shape: (dimt, dimx, numVecs)

def buildPerambulator(modelObj: schwingerModel, configIndex: int, numVecs: int, chemicalPot=0):
    """
    Computes the distillation perambulator for a single gauge configuration.

    Returns tau of shape (dimt, dimt, numVecs*2, numVecs*2)
      tau[t_sink, t_src, l*2+s_sink, k*2+s_src]
        = sum_x V(t_sink)[x,l]* M^{-1}[x,t_sink,s_sink; x',t_src,s_src] V(t_src)[x',k]
    """
    gaugeLinks = modelObj.linkHistory[configIndex]
    eigVecs = findPartialEigenBasis(modelObj, configIndex, numVecs)

    # eigVecs shape: (dimt, dimx, numVecs)

    N_t, N_x, N_vec = eigVecs.shape

    lu = splu(buildDiracOp(modelObj, gaugeLinks, chemicalPot).tocsc())

    tau = np.zeros((N_t, N_t, N_vec*2, N_vec*2), dtype=complex)

    for t_src in range(N_t):
        # Build sources: one column per (k, s), localized at t_src
        B = np.zeros((N_x*N_t*2, N_vec*2), dtype=complex)
        for s in range(2):
            rows = np.arange(N_x)*N_t*2 + t_src*2 + s
            B[np.ix_(rows, np.arange(N_vec)*2 + s)] = eigVecs[t_src]  # (N_x, N_vec)

        Phi = lu.solve(B).reshape(N_x, N_t, 2, N_vec, 2)
        # (x, t_sink, s_sink, k_src, s_src)

        # einsum: t=t_sink, a=x (contracted), i=l_sink, j=s_sink, k=k_src, d=s_src
        # compound row index: l*2+s_sink, compound col index: k*2+s_src
        tau[:, t_src] = np.einsum('tai, atjkd -> tijkd', eigVecs.conj(), Phi, optimize=True).reshape(N_t, N_vec*2, N_vec*2)

    return tau

def covariantDerivative(modelObj: schwingerModel, gaugeLinks, nt):
    """
    Symmetric gauge-covariant spatial derivative at time slice nt (no spin index):
        (D psi)(x) = [U_x(x) psi(x+1) - U_x^*(x-1) psi(x-1)] / (2a)
    """
    shift_x_1Dpos = np.roll(np.eye(modelObj.dimx), -1, axis=0) # This is \delta_{x_n+1, x_m}
    shift_x_1Dneg = np.roll(np.eye(modelObj.dimx), +1, axis=0) # This is \delta_{x_n-1, x_m}

    #flattened gaugelinks: [:,nt, 1] are spatial links at timeslice t
    spaceLinks = sparse.diags_array(gaugeLinks[:,nt,1].flatten())

    D = (spaceLinks@shift_x_1Dpos - shift_x_1Dneg@(spaceLinks.conj()))/(2*modelObj.a)

    return D

def derivativeKernel(numDerivs):
    """
    Returns a spatial kernel function K(modelObj, gaugeLinks, nt) = D^numDerivs
    for use in buildElemental. numDerivs=0 gives the identity.
    NOTE: odd numbers of derivatives flip the spatial parity of the operator, so at
    zero momentum their cross-correlators with even-derivative operators vanish.
    """
    def kernel(modelObj, gaugeLinks, nt):
        K = sparse.eye_array(modelObj.dimx)
        for _ in range(numDerivs):
            K = covariantDerivative(modelObj, gaugeLinks, nt) @ K
        return K
    return kernel

def buildElemental(modelObj: schwingerModel, configIndex:int, numVecs: int, kernel=None,
                    gamma=np.array([[1j,0],[0,-1j]]), momk=0, eigVecs=None):
    """
    Builds the meson elemental for one configuration:
        Phi[t] = kron( V(t)^dag e^{-i 2pi momk x/L} K(t) V(t), gamma )
    shape (dimt, numVecs*2, numVecs*2) with the compound index l*2+s matching
    buildPerambulator. kernel=None means the identity spatial kernel, for which
    (at momk=0) orthonormality of V gives Phi[t] = kron(eye(numVecs), gamma).
    """
    gaugeLinks = modelObj.linkHistory[configIndex]
    if eigVecs is None:
        eigVecs = findPartialEigenBasis(modelObj, configIndex, numVecs)

    N_t, N_x, N_vec = eigVecs.shape

    #momentum projection phase, applied on the psibar side
    phase = np.exp(-1j*2*np.pi*momk*np.arange(N_x)/N_x)

    elemental = np.zeros((N_t, N_vec*2, N_vec*2), dtype=complex)
    for nt in range(N_t):
        if kernel is None:
            Kv = eigVecs[nt]
        else:
            Kv = kernel(modelObj, gaugeLinks, nt) @ eigVecs[nt]
        vkv = eigVecs[nt].conj().T @ (phase[:,None] * Kv)
        elemental[nt] = np.kron(vkv, gamma)

    return elemental

def operatorBasis(maxDerivs=2, gammas=None):
    """
    Standard operator basis for GEVP: every gamma structure paired with
    0..maxDerivs covariant derivatives. gammas is a dict {name: (2,2) array},
    defaulting to the pseudoscalar used everywhere else in this repo.
    """
    if gammas is None:
        gammas = {"g5": np.array([[1j,0],[0,-1j]])}

    ops = []
    for gName, gamma in gammas.items():
        for nD in range(maxDerivs+1):
            kernel = derivativeKernel(nD) if nD > 0 else None
            ops.append(wick.mesonOp(f"{gName}_D{nD}", gamma, kernel))
    return ops

class distillationSpace:
    """
    Manages the distillation objects for one model: eigenvector bases,
    perambulators, and elementals are computed lazily and cached per
    configuration (and per operator for the elementals).
    """

    def __init__(self, modelObj: schwingerModel, numVecs: int, chemicalPot=0):
        self.modelObj = modelObj
        self.numVecs = numVecs
        self.chemicalPot = chemicalPot

        self._eigenBases = {}
        self._perambulators = {}
        self._elementals = {}   # keyed by (configIndex, op.name)

    def eigenBasis(self, configIndex):
        if configIndex not in self._eigenBases:
            self._eigenBases[configIndex] = findPartialEigenBasis(self.modelObj, configIndex, self.numVecs)
        return self._eigenBases[configIndex]

    def perambulator(self, configIndex):
        if configIndex not in self._perambulators:
            self._perambulators[configIndex] = buildPerambulator(self.modelObj, configIndex,
                                                                 self.numVecs, chemicalPot=self.chemicalPot)
        return self._perambulators[configIndex]

    def elemental(self, configIndex, op):
        """op is anything with .name/.gamma/.kernel/.momk (mesonOp or bilinear)."""
        if (configIndex, op.name) not in self._elementals:
            self._elementals[(configIndex, op.name)] = buildElemental(
                self.modelObj, configIndex, self.numVecs, kernel=op.kernel,
                gamma=op.gamma, momk=getattr(op, 'momk', 0),
                eigVecs=self.eigenBasis(configIndex))
        return self._elementals[(configIndex, op.name)]

    def elementalBar(self, configIndex, op):
        """
        Elemental of the daggered operator:
            (psibar G e^{-ikx} K psi)^dag = psibar Gbar K^dag e^{+ikx} psi
        with Gbar = gammat G^dag gammat, so PhiBar[t] = Phi[t]^dag conjugated by gammat
        (the momentum flip comes along with the dagger of the spatial part).
        """
        key = (configIndex, op.name + "__bar")
        if key not in self._elementals:
            gt = np.kron(np.eye(self.numVecs), self.modelObj.gammat)
            phi = self.elemental(configIndex, op)
            self._elementals[key] = gt @ phi.conj().transpose(0,2,1) @ gt
        return self._elementals[key]

def getElementalCorrelMatrix(modelObj: schwingerModel, configIndex: int, numVecs: int, ops,
                              chemicalPot=0):
    """
    Per-configuration pieces of the correlation matrix C_ij(t) = <O_i(t) O_j(0)^dag>
    for a list of mesonOps (flavor coefficients are applied later by the driver).

    Returns (conn, loopsSnk, loopsSrc):
        conn[i,j,dt]  = (1/T) sum_t tr[Phi_i(t+dt) tau(t+dt,t) PhiBar_j(t) tau(t,t+dt)]
        loopsSnk[i,t] = tr[Phi_i(t) tau(t,t)]
        loopsSrc[j,t] = tr[PhiBar_j(t) tau(t,t)]
    """
    space = distillationSpace(modelObj, numVecs, chemicalPot=chemicalPot)
    tau = space.perambulator(configIndex)
    tauDiag = np.einsum('ttij->tij', tau)

    nOps = len(ops)
    dimt = modelObj.dimt

    conn = np.zeros((nOps, nOps, dimt), dtype=complex)
    loopsSnk = np.zeros((nOps, dimt), dtype=complex)
    loopsSrc = np.zeros((nOps, dimt), dtype=complex)

    for i, opSnk in enumerate(ops):
        phiSnk = space.elemental(configIndex, opSnk)
        loopsSnk[i] = np.einsum('tab,tba->t', phiSnk, tauDiag, optimize=True)

        for j, opSrc in enumerate(ops):
            phiBarSrc = space.elementalBar(configIndex, opSrc)
            if i == 0:
                loopsSrc[j] = np.einsum('tab,tba->t', phiBarSrc, tauDiag, optimize=True)

            # trace[t_snk, t_src] = tr[Phi_i(t_snk) tau(t_snk,t_src) PhiBar_j(t_src) tau(t_src,t_snk)]
            trace = np.einsum('tab,tsbc,scd,stda->ts', phiSnk, tau, phiBarSrc, tau, optimize=True)

            #average over source position at fixed separation
            conn[i,j] = np.array([
                np.roll(trace, -dt, axis=0).diagonal().mean()
                for dt in range(dimt)
            ])

    return conn, loopsSnk, loopsSrc

def pionBasis(derivCounts=(0, 2), pipiMomenta=(1,), gamma=None):
    """
    GEVP basis of interpOps that all share the pion quantum numbers
    (I=1, I3=+1, P=-1, total momentum 0):
      - single pions psibar_d [g5 D^n] psi_u for each n in derivCounts
        (only EVEN n keeps P=-1; odd n is a different channel)
      - two-pion operators pi(k)pi(-k) in the antisymmetric I=1 combination
        for each k in pipiMomenta
    """
    ops = []
    for nD in derivCounts:
        kernel = derivativeKernel(nD) if nD > 0 else None
        ops.append(wick.singleMesonOp(f"pi_D{nD}", wick.PION_PLUS, gamma, kernel))
    for k in pipiMomenta:
        ops.append(wick.piPiOpI1(k, gamma))
    return ops

def _cycleTrace(phis, tau, timeSyms):
    """
    One fermion loop of a Wick contraction:
        Tr[ Phi_1(T_1) tau(T_1,T_2) Phi_2(T_2) tau(T_2,T_3) ... tau(T_k,T_1) ]
    phis: list of (dimt, N, N) elementals along the cycle
    timeSyms: 's' (sink time) or 'r' (source time) for each vertex
    Returns the trace as an array over the distinct times present:
    shape (dimt, dimt) as (t_snk, t_src) if both appear, else (dimt,).
    """
    letters = 'abcdefghijklmnop'
    k = len(phis)

    subs = []
    operands = []
    for j in range(k):
        row, col = letters[2*j], letters[2*j+1]
        nxt = letters[(2*j+2) % (2*k)]
        subs.append(timeSyms[j] + row + col)
        operands.append(phis[j])
        subs.append(timeSyms[j] + timeSyms[(j+1) % k] + col + nxt)
        operands.append(tau)

    out = ('s' if 's' in timeSyms else '') + ('r' if 'r' in timeSyms else '')

    return np.einsum(','.join(subs) + '->' + out, *operands, optimize=True)

def getInterpCorrelMatrix(modelObj: schwingerModel, configIndex: int, numVecs: int, ops,
                           chemicalPot=0):
    """
    Per-configuration correlation matrix for general interpOps (sums of products
    of bilinears, e.g. mixed single-meson / multi-meson bases):
        C[i,j,dt] = (1/T) sum_t  [all Wick contractions of O_i(t+dt) O_j(t)^dag]
    Contractions are generated by the permutation method (wick.wickContractions)
    on the flavor labels; every cycle becomes a trace over elementals and
    perambulators via _cycleTrace, and the contraction carries (-1)^{#loops}.

    Also returns per-op VEV series (vevSnk[i,t] = <O_i(t)>_config and
    vevSrc[j,t] = <O_j(t)^dag>_config) so the driver can do ensemble-level
    vacuum subtraction; these vanish identically for I != 0 channels.
    """
    space = distillationSpace(modelObj, numVecs, chemicalPot=chemicalPot)
    tau = space.perambulator(configIndex)

    nOps = len(ops)
    dimt = modelObj.dimt

    cycleCache = {}

    def cycleVal(verts, cycle):
        #verts: list of (bilinear, timeSym, barred); cycle: tuple of vertex indices
        #canonicalize under rotation so equivalent loops share a cache entry
        labels = tuple((verts[i][0].name, verts[i][1], verts[i][2]) for i in cycle)
        rotations = [labels[r:] + labels[:r] for r in range(len(labels))]
        key = min(rotations)

        if key not in cycleCache:
            rot = rotations.index(key)
            order = cycle[rot:] + cycle[:rot]
            phis = [space.elementalBar(configIndex, verts[i][0]) if verts[i][2]
                    else space.elemental(configIndex, verts[i][0]) for i in order]
            timeSyms = [verts[i][1] for i in order]
            cycleCache[key] = (_cycleTrace(phis, tau, timeSyms), ''.join(sorted(set(timeSyms))))
        return cycleCache[key]

    def contractTerms(verts):
        #sum of all Wick contractions as an array over (t_snk, t_src)
        #effective (barFlavor, flavor) of each vertex: daggering swaps them
        eff = [(b.flavor, b.barFlavor) if barred else (b.barFlavor, b.flavor)
               for b, ts, barred in verts]

        acc = np.zeros((dimt, dimt), dtype=complex)
        for sign, cycles in wick.wickContractions(eff):
            val = np.ones((dimt, dimt), dtype=complex)
            for cycle in cycles:
                arr, syms = cycleVal(verts, cycle)
                if syms == 'rs':
                    val = val * arr
                elif syms == 's':
                    val = val * arr[:, None]
                else:
                    val = val * arr[None, :]
            acc += sign * val
        return acc

    C = np.zeros((nOps, nOps, dimt), dtype=complex)
    vevSnk = np.zeros((nOps, dimt), dtype=complex)
    vevSrc = np.zeros((nOps, dimt), dtype=complex)

    for i, opSnk in enumerate(ops):
        for j, opSrc in enumerate(ops):
            full = np.zeros((dimt, dimt), dtype=complex)
            for cS, bsS in opSnk.terms:
                for cB, bsB in opSrc.terms:
                    #sink bilinears at t_snk, daggered source bilinears at t_src
                    verts = [(b, 's', False) for b in bsS] + [(b, 'r', True) for b in bsB]
                    full += cS*np.conj(cB) * contractTerms(verts)

            #average over source position at fixed separation
            C[i,j] = np.array([
                np.roll(full, -dt, axis=0).diagonal().mean()
                for dt in range(dimt)
            ])

    #per-op VEV series for ensemble-level vacuum subtraction
    for i, op in enumerate(ops):
        for c, bs in op.terms:
            vertsS = [(b, 's', False) for b in bs]
            vertsB = [(b, 's', True) for b in bs]
            vevSnk[i] += c * contractTerms(vertsS).diagonal()
            vevSrc[i] += np.conj(c) * contractTerms(vertsB).diagonal()

    return C, vevSnk, vevSrc

def interpGEVPStats(modelObj: schwingerModel, ops, burnIn=1, autocorrSkip=1, numVecs=4,
                    chemicalPot=0, theta=0, ti=1, numResamples=2000,
                    vacuumSubtract=True, thermalShift=False, n_jobs=-1):
    """
    Distillation GEVP over a basis of general interpOps (single mesons with any
    gamma / derivative kernel / momentum, and multi-meson operators) that share
    the same quantum numbers, e.g. pionBasis(). All Wick contractions -- connected,
    disconnected, and the box/triangle diagrams of multi-particle operators --
    are generated automatically by the permutation method.

    thermalShift: if True, the GEVP is solved on the shifted matrix
        Ctilde(t) = C(t) - C(t+1)
    which removes time-constant thermal (around-the-world) contamination, e.g.
    the pi(fwd)pi(bwd) piece of two-pion correlators. The eigenvalue correlators
    then decay as pure exponentials -- fit with an exp, not a cosh -- and the
    time extent of the output shrinks by one.

    Returns [mean (dimtEff-ti, nOps), errors (2, dimtEff-ti, nOps),
    covMat (nOps, dimtEff-ti, dimtEff-ti), refVecs (nOps, nOps)] with
    dimtEff = dimt-1 if thermalShift else dimt, compatible with
    correlation.gevpMassExtract. refVecs column k is state k's eigenvector:
    its operator content in the basis (normalized in the C(ti) metric).
    """
    from .correlation import gevp

    nOps = len(ops)
    dimt = modelObj.dimt

    weights = getWeightingFactorsTheta(modelObj, theta=theta, burnIn=burnIn, autocorrSkip=autocorrSkip)
    indices = np.arange(burnIn, modelObj.metroSteps, autocorrSkip)

    with tqdm_joblib(tqdm(total=len(indices), desc="Interp. configs")):
        perConfig = Parallel(n_jobs=n_jobs)(
            delayed(getInterpCorrelMatrix)(modelObj, i, numVecs, ops, chemicalPot=chemicalPot)
            for i in indices)

    C      = np.array([r[0] for r in perConfig])   # (n_configs, nOps, nOps, dimt)
    vevSnk = np.array([r[1] for r in perConfig])   # (n_configs, nOps, dimt)
    vevSrc = np.array([r[2] for r in perConfig])

    def _meanMatrix(idx=None):
        #weighted-mean correlation matrix with vacuum subtraction;
        #idx indexes bootstrap resamples
        if idx is None:
            c, vS, vB, w = C, vevSnk, vevSrc, weights
        else:
            c, vS, vB, w = C[idx], vevSnk[idx], vevSrc[idx], weights[idx]

        ws = np.sum(w, axis=-1)
        Cm = np.sum(c * w[..., None, None, None], axis=-4) / ws[..., None, None, None]

        if vacuumSubtract:
            vSm = np.sum(vS * w[..., None, None], axis=-3) / ws[..., None, None]  # <O_i(t)>
            vBm = np.sum(vB * w[..., None, None], axis=-3) / ws[..., None, None]  # <O_j(t)^dag>
            # vac[i,j,dt] = (1/T) sum_t <O_i(t+dt)> <O_j(t)^dag>
            vac = np.stack([
                np.einsum('...it,...jt->...ij', np.roll(vSm, -dt, axis=-1), vBm)/dimt
                for dt in range(dimt)
            ], axis=-1)
            Cm = Cm - vac
        return Cm

    def _gevpEigs(Cm, returnVecs=False):
        # Cm: (nOps, nOps, dimt) -> state-tracked eigenvalue correlators (dimtEff-ti, nOps)
        corrMat = np.real((Cm + np.conj(np.swapaxes(Cm, 0, 1)))/2)  #hermitize
        if thermalShift:
            corrMat = corrMat[:, :, :-1] - corrMat[:, :, 1:]
        newCorrs, refVecs = gevp(corrMat, ti=ti)
        if returnVecs:
            return np.real(newCorrs), refVecs
        return np.real(newCorrs)

    totalCorrelMean, refVecs = _gevpEigs(_meanMatrix(), returnVecs=True)

    dimtEff = dimt - 1 if thermalShift else dimt

    #chunked bootstrap over configurations
    chunk_size = 500
    rng = np.random.default_rng()

    bootstrap_means = np.empty((numResamples, dimtEff - ti, nOps))

    for start in range(0, numResamples, chunk_size):
        end = min(start + chunk_size, numResamples)
        idx = rng.choice(len(C), size=(end - start, len(C)))

        C_chunk = _meanMatrix(idx)
        for s in range(end - start):
            bootstrap_means[start + s] = _gevpEigs(C_chunk[s])
        del C_chunk

    low  = np.percentile(bootstrap_means, 2.5,  axis=0)
    high = np.percentile(bootstrap_means, 97.5, axis=0)

    covMat = np.array([np.cov(bootstrap_means[:, :, eigIdx], rowvar=False) for eigIdx in range(nOps)])

    return [totalCorrelMean, np.array([high - totalCorrelMean, totalCorrelMean - low]), covMat, refVecs]

def getCorrelation(modelObj: schwingerModel, configIndex: int, numVecs: int, chemicalPot=0,
                    gamma=np.array([[1j,0],[0,-1j]])):

    peramb = buildPerambulator(modelObj, configIndex, numVecs, chemicalPot=chemicalPot)

    elemental = np.kron(np.eye(numVecs),gamma)

    trace = -np.einsum("ijkl,lm,jimn,nk->ij",peramb,elemental,peramb,elemental,optimize=True)

    correlator = np.array([
        np.roll(trace, -dt, axis=0).diagonal().mean()
        for dt in range(modelObj.dimt)
    ])

    return correlator

def getCorrelationLoop(modelObj: schwingerModel, configIndex: int, numVecs: int, chemicalPot=0,
                    gamma=np.array([[1j,0],[0,-1j]])):
    #assuming isospin symmetry for everything
    
    peramb = buildPerambulator(modelObj, configIndex, numVecs, chemicalPot=chemicalPot)

    elemental = np.kron(np.eye(numVecs),gamma)

    trace = np.einsum("iikl,lk->i",peramb,elemental,optimize=True)

    return trace

def correlStats(modelObj: schwingerModel, burnIn=1, autocorrSkip=1,
                    Gamma=np.array([[1j,0],[0,-1j]]), nVec=2, chemicalPot=0, theta=0,disc=False):
    
    weights = getWeightingFactorsTheta(modelObj, theta=theta,burnIn=burnIn, autocorrSkip=autocorrSkip)
    
    indices = np.arange(burnIn, modelObj.metroSteps, autocorrSkip)
    with tqdm_joblib(tqdm(total=len(indices), desc="Conn. configs")):
        correl = np.array(Parallel(n_jobs=-1)(delayed(getCorrelation)(modelObj, i, nVec, gamma=Gamma,chemicalPot=chemicalPot) for i in indices))

    if(disc):
        with tqdm_joblib(tqdm(total=len(indices), desc="Disc. Loops")):
            loops = np.array(Parallel(n_jobs=-1)(delayed(getCorrelationLoop)(modelObj, i, nVec, gamma=Gamma,chemicalPot=chemicalPot) for i in indices))
        # loops shape: (n_configs, dimt), loops[n,t] = L_n(t) = Tr[Phi tau(t,t)]

    totalCorrelMean = np.real(np.average(correl,axis=0,weights=weights))

    if(disc):
        dimt = modelObj.dimt

        # per-config, translation-averaged loop-loop product on the SAME config:
        #   loopCorrel[n, dt] = (1/T) sum_t L_n(t+dt) L_n(t)
        loopCorrel = np.stack([
            np.mean(np.roll(loops, -dt, axis=1) * loops, axis=1)
            for dt in range(dimt)
        ], axis=1)                                       # (n_configs, dimt)

        def _discCorrel(llc, lp, w):
            # llc, lp: (..., n_configs, dimt); w: (..., n_configs)
            # returns vacuum-subtracted disconnected correlator (..., dimt)
            ws   = np.sum(w, axis=-1, keepdims=True)
            LL   = np.sum(llc * w[..., None], axis=-2) / ws          # <(1/T) sum_t L(t+dt)L(t)>
            Lbar = np.sum(lp  * w[..., None], axis=-2) / ws          # <L(t)>  (..., dimt)
            vac  = np.stack([                                        # (1/T) sum_t <L(t+dt)><L(t)>
                np.mean(np.roll(Lbar, -dt, axis=-1) * Lbar, axis=-1)
                for dt in range(dimt)
            ], axis=-1)
            return LL - vac

        totalCorrelMean = totalCorrelMean + np.real(_discCorrel(loopCorrel, loops, weights))

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
            chunk_means = chunk_means + np.real(
                _discCorrel(loopCorrel[idx], loops[idx], w_chunk)
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


def _discCorrelPair(loopCorrel, loopsSnk, loopsSrc, w):
    """
    Vacuum-subtracted disconnected correlator from two (possibly different) loops.
    loopCorrel: (..., n_configs, dimt) per-config translation-averaged LsnkLsrc product
    loopsSnk, loopsSrc: (..., n_configs, dimt); w: (..., n_configs)
    Returns (..., dimt).
    """
    dimt = loopsSnk.shape[-1]
    ws    = np.sum(w, axis=-1, keepdims=True)
    LL    = np.sum(loopCorrel * w[..., None], axis=-2) / ws     # <(1/T) sum_t Lsnk(t+dt)Lsrc(t)>
    LbarS = np.sum(loopsSnk   * w[..., None], axis=-2) / ws     # <Lsnk(t)>
    LbarB = np.sum(loopsSrc   * w[..., None], axis=-2) / ws     # <Lsrc(t)>
    vac   = np.stack([                                          # (1/T) sum_t <Lsnk(t+dt)><Lsrc(t)>
        np.mean(np.roll(LbarS, -dt, axis=-1) * LbarB, axis=-1)
        for dt in range(dimt)
    ], axis=-1)
    return LL - vac


def distillGEVPStats(modelObj: schwingerModel, ops=None, flavorTerms=wick.PION_PLUS,
                     burnIn=1, autocorrSkip=1, numVecs=4, chemicalPot=0, theta=0,
                     ti=1, numResamples=2000, thermalShift=False, n_jobs=-1):
    """
    Distillation GEVP over a basis of mesonOps (different gammas / covariant
    derivatives): builds the full correlation matrix C_ij(t) = <O_i(t) O_j(0)^dag>
    including the disconnected diagrams required by the channel's flavor structure
    (wick.PION_PLUS / PION_ZERO: connected only, wick.ETA: connected + 2x disc.),
    then solves the GEVP on the ensemble mean and on each bootstrap resample.

    Returns [mean (dimt-ti, nOps), errors (2, dimt-ti, nOps), covMat (nOps, dimt-ti, dimt-ti),
    refVecs (nOps, nOps)], compatible with correlation.gevpMassExtract.
    refVecs column k is state k's eigenvector: its operator content in the basis.
    """
    from .correlation import gevp

    if ops is None:
        ops = operatorBasis(maxDerivs=2)
    nOps = len(ops)
    dimt = modelObj.dimt

    #wick contraction coefficients from the permutation method
    connCoeff, discCoeff = wick.twoPointCoeffs(flavorTerms)

    weights = getWeightingFactorsTheta(modelObj, theta=theta, burnIn=burnIn, autocorrSkip=autocorrSkip)
    indices = np.arange(burnIn, modelObj.metroSteps, autocorrSkip)

    with tqdm_joblib(tqdm(total=len(indices), desc="Distill. configs")):
        perConfig = Parallel(n_jobs=n_jobs)(
            delayed(getElementalCorrelMatrix)(modelObj, i, numVecs, ops, chemicalPot=chemicalPot)
            for i in indices)

    conn     = connCoeff * np.array([r[0] for r in perConfig])   # (n_configs, nOps, nOps, dimt)
    loopsSnk = np.array([r[1] for r in perConfig])               # (n_configs, nOps, dimt)
    loopsSrc = np.array([r[2] for r in perConfig])

    disc = (discCoeff != 0)
    if disc:
        # per-config translation-averaged loop-loop product on the SAME config:
        #   loopCorrel[n, i, j, dt] = (1/T) sum_t Lsnk_i(t+dt) Lsrc_j(t)
        loopCorrel = np.stack([
            np.einsum('nit,njt->nij', np.roll(loopsSnk, -dt, axis=-1), loopsSrc, optimize=True)/dimt
            for dt in range(dimt)
        ], axis=-1)                                              # (n_configs, nOps, nOps, dimt)

    def _meanMatrix(idx=None):
        #weighted-mean correlation matrix; idx indexes bootstrap resamples
        if idx is None:
            c, w = conn, weights
            lC = loopCorrel if disc else None
            lS, lB = (loopsSnk, loopsSrc) if disc else (None, None)
        else:
            c, w = conn[idx], weights[idx]
            lC = loopCorrel[idx] if disc else None
            lS, lB = (loopsSnk[idx], loopsSrc[idx]) if disc else (None, None)

        wExp = w[..., None, None, None]
        C = np.sum(c * wExp, axis=-4) / np.sum(wExp, axis=-4)

        if disc:
            #disc piece for every (i,j): loop over ops (nOps is small)
            for i in range(nOps):
                for j in range(nOps):
                    C[..., i, j, :] = C[..., i, j, :] + discCoeff * _discCorrelPair(
                        lC[..., :, i, j, :], lS[..., :, i, :], lB[..., :, j, :], w)
        return C

    def _gevpEigs(C, returnVecs=False):
        # C: (nOps, nOps, dimt) -> state-tracked eigenvalue correlators (dimtEff-ti, nOps)
        corrMat = np.real((C + np.conj(np.swapaxes(C, 0, 1)))/2)  #hermitize
        if thermalShift:
            corrMat = corrMat[:, :, :-1] - corrMat[:, :, 1:]
        newCorrs, refVecs = gevp(corrMat, ti=ti)
        if returnVecs:
            return np.real(newCorrs), refVecs
        return np.real(newCorrs)

    totalCorrelMean, refVecs = _gevpEigs(_meanMatrix(), returnVecs=True)  # (dimtEff-ti, nOps)

    dimtEff = dimt - 1 if thermalShift else dimt

    #chunked bootstrap over configurations
    chunk_size = 500
    rng = np.random.default_rng()

    bootstrap_means = np.empty((numResamples, dimtEff - ti, nOps))

    for start in range(0, numResamples, chunk_size):
        end = min(start + chunk_size, numResamples)
        idx = rng.choice(len(conn), size=(end - start, len(conn)))

        C_chunk = _meanMatrix(idx)                                # (chunk, nOps, nOps, dimt)
        for s in range(end - start):
            bootstrap_means[start + s] = _gevpEigs(C_chunk[s])
        del C_chunk

    low  = np.percentile(bootstrap_means, 2.5,  axis=0)           # (dimt-ti, nOps)
    high = np.percentile(bootstrap_means, 97.5, axis=0)

    # Per-eigenvalue covariance: (nOps, dimt-ti, dimt-ti)
    covMat = np.array([np.cov(bootstrap_means[:, :, eigIdx], rowvar=False) for eigIdx in range(nOps)])

    return [totalCorrelMean, np.array([high - totalCorrelMean, totalCorrelMean - low]), covMat, refVecs]


