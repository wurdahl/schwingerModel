from functools import partial
import os

#CUDA graphs (XLA command buffers): the small-lattice HMC is kernel-launch-latency
#bound, so recording launch sequences as graphs and replaying them cuts per-kernel
#overhead. WHILE lets the CG loops themselves be captured; min_graph_size=2 captures
#even short sequences. Must be set before jax initializes; setdefault means an
#XLA_FLAGS already present in the environment wins untouched (also the escape hatch:
#XLA_FLAGS="" disables this).
os.environ.setdefault(
    'XLA_FLAGS',
    '--xla_gpu_enable_command_buffer=FUSION,CUBLAS,CUBLASLT,CUDNN,CUSTOM_CALL,CONDITIONAL,WHILE'
    ' --xla_gpu_graph_min_graph_size=2')

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
def cg(A, b, x0=None, tol=1e-5, maxiter=None):
    """Conjugate gradient for Hermitian positive-definite matvecs.

    Same algorithm and stopping rule as jax.scipy.sparse.linalg.cg, without its
    custom_linear_solve autodiff wrapper — that wrapper requires a transpose rule
    for every op in the matvec, which the CUDA FFI operator kernel does not have
    (and none of our solves are ever differentiated through). Returns (x, None)
    to keep the (x, info) call shape.
    """
    if maxiter is None:
        maxiter = 10 * b.size
    x0 = jnp.zeros_like(b) if x0 is None else x0
    #stop when |r|^2 <= (tol*|b|)^2, matching the jax/scipy default criterion
    atol2 = tol * tol * jnp.vdot(b, b).real

    r0 = b - A(x0)
    rs0 = jnp.vdot(r0, r0).real

    def cond(state):
        _, _, _, rs, k = state
        return (rs > atol2) & (k < maxiter)

    def body(state):
        x, r, p, rs, k = state
        Ap = A(p)
        alpha = rs / jnp.vdot(p, Ap).real   #real for Hermitian positive-definite A
        x = x + alpha * p
        r = r - alpha * Ap
        rsNew = jnp.vdot(r, r).real
        p = r + (rsNew / rs) * p
        return x, r, p, rsNew, k + 1

    x, _, _, _, _ = jax.lax.while_loop(cond, body, (x0, r0, r0, rs0, 0))
    return x, None

import numpy as np
from tqdm import tqdm

from .params import dwfParams

#fused CUDA operator kernel (cuda/dwfApplyD.cu, built by cuda/build.sh): one
#kernel launch per apply instead of ~25. Optional: falls back to the XLA roll
#implementation when the library isn't built. SCHWINGER_NO_CUDA_KERNEL=1
#disables it for A/B comparisons.
try:
    from . import dwf_cuda as _dwf_cuda
    _USE_CUDA_KERNEL = _dwf_cuda.available() and os.environ.get("SCHWINGER_NO_CUDA_KERNEL") != "1"
except Exception:
    _USE_CUDA_KERNEL = False

#spin matrices as jax constants (same convention as params.py)
_GAMMA_X = jnp.array([[0, 1], [1, 0]], dtype=jnp.complex128)
_GAMMA_T = jnp.array([[0, -1j], [1j, 0]], dtype=jnp.complex128)
_GAMMA_5 = jnp.array([[1, 0], [0, -1]], dtype=jnp.complex128)


#settings is static: shapes depend on it, and dwfParams is hashable so the jit
#cache keys on it (a different fMass or lattice size compiles its own version)
@partial(jax.jit, static_argnums=0)
def pseudoBilinear(settings, pseudoField, gaugeLinks, cgRtol):

    operator = lambda v: applyD_dwf(settings, gaugeLinks, applyDdag_dwf(settings, gaugeLinks, v))

    X, _ = cg(operator, pseudoField, tol=cgRtol)
    relRes = jnp.linalg.norm(operator(X) - pseudoField) / jnp.linalg.norm(pseudoField)

    return jnp.vdot(pseudoField, X).real, relRes


def fermionForceBilinear(settings: dwfParams, gaugeLinks, left, right):
    """2*Re <left| dD/dtheta |right> at each link, summed over the s5 slices.

    The gauge links enter D_dwf only through the Wilson term, replicated on every
    s5 slice, so dD/dtheta is the same per-slice expression as the Wilson case
    (and the same for any fMass, so the Pauli-Villars force can reuse this).
    left, right: shape (dim5, dimx, dimt, 2). Returns real (dimx, dimt, 2).
    """
    #constants follow the field dtype so the c64 force path stays c64
    dataType = right.dtype
    realType = jnp.zeros((), dataType).real.dtype
    I = jnp.eye(2, dtype=dataType)
    c = 1j / (2 * settings.a)

    Ux = gaugeLinks[:, :, 1]  # (dimx, dimt)
    Ut = gaugeLinks[:, :, 0]  # (dimx, dimt)

    P_minus_x = I - _GAMMA_X.astype(dataType)
    P_plus_x  = I + _GAMMA_X.astype(dataType)
    P_minus_t = I - _GAMMA_T.astype(dataType)
    P_plus_t  = I + _GAMMA_T.astype(dataType)

    #axis 0 is s5, axis 1 is x, axis 2 is t
    R_xp1 = jnp.roll(right, shift=-1, axis=1)  # right[s, x+1, t, :]
    L_xp1 = jnp.roll(left,  shift=-1, axis=1)  # left[s, x+1, t, :]
    R_tp1 = jnp.roll(right, shift=-1, axis=2)  # right[s, x, t+1, :]
    L_tp1 = jnp.roll(left,  shift=-1, axis=2)  # left[s, x, t+1, :]


    # Spatial: Z_x = -c*Ux * <L|P-_x|R_{x+1}> + c*Ux* * <L_{x+1}|P+_x|R>
    Pm_x_R_xp1 = jnp.einsum('ij,sxyj->sxyi', P_minus_x, R_xp1)
    Pp_x_R     = jnp.einsum('ij,sxyj->sxyi', P_plus_x,  right)
    Z_x = (-c * Ux      * jnp.einsum('sxyi,sxyi->xy', jnp.conj(left),  Pm_x_R_xp1)
            + c * jnp.conj(Ux) * jnp.einsum('sxyi,sxyi->xy', jnp.conj(L_xp1), Pp_x_R))
    

    # Time: Z_t = -c*Ut * <L|P-_t|R_{t+1}> + c*Ut* * <L_{t+1}|P+_t|R>
    Pm_t_R_tp1 = jnp.einsum('ij,sxyj->sxyi', P_minus_t, R_tp1)
    Pp_t_R     = jnp.einsum('ij,sxyj->sxyi', P_plus_t,  right)
    Z_t = (-c * Ut      * jnp.einsum('sxyi,sxyi->xy', jnp.conj(left),  Pm_t_R_tp1)
            + c * jnp.conj(Ut) * jnp.einsum('sxyi,sxyi->xy', jnp.conj(L_tp1), Pp_t_R))

    # Anti-periodic boundary condition: flip sign at t = dimt-1
    bc_t = jnp.ones((settings.dimx, settings.dimt), dtype=realType).at[:, -1].set(-1.0)
    
    return jnp.stack([2 * Z_t.real * bc_t,    # time links  -> out[:, :, 0]
                      2 * Z_x.real], axis=-1) # space links -> out[:, :, 1]


def pvForce(settings: dwfParams, gaugeLinks, chi):
    """Force from the Pauli-Villars action S_PV = chi^dag D(1)^dag D(1) chi.

    No inverse in the action means no CG solve, and no minus sign:
    dS_PV = +2*Re<D(1)chi| dD |chi>.
    """

    W = applyD_dwf(settings._replace(fMass=1.0),gaugeLinks,chi)

    return fermionForceBilinear(settings, gaugeLinks, W, chi)


@partial(jax.jit, static_argnums=0)
def hmcForcingFunction_vec(settings: dwfParams, gaugeLinks, phis, chi, x0=None, cgRtol=1e-5):

    gaugeLinks = gaugeLinks.astype(jnp.complex64)
    phis = phis.astype(jnp.complex64)
    chi = chi.astype(jnp.complex64)

    operator = lambda v: applyD_dwf(settings, gaugeLinks, applyDdag_dwf(settings, gaugeLinks, v))
    
    X, _ = cg(operator, phis, tol=cgRtol,x0=x0)

    #Y = D^\dagger X
    Y = applyDdag_dwf(settings, gaugeLinks, X)

    Ux = gaugeLinks[:, :, 1]  # (dimx, dimt)
    Ut = gaugeLinks[:, :, 0]  # (dimx, dimt)

    # --- Gauge force (vectorized staples) ---
    Ux_tp1     = jnp.roll(Ux, shift=-1, axis=1)              # Ux[x, t+1]
    Ut_xp1     = jnp.roll(Ut, shift=-1, axis=0)              # Ut[x+1, t]
    Ux_xm1     = jnp.roll(Ux, shift=1,  axis=0)              # Ux[x-1, t]
    Ux_xm1_tp1 = jnp.roll(Ux_tp1, shift=1, axis=0)          # Ux[x-1, t+1]
    Ut_xm1     = jnp.roll(Ut, shift=1,  axis=0)              # Ut[x-1, t]

    # Time links: right staple = Ux[x,t+1]*Ut*[x+1,t]*Ux*[x,t]
    #             left staple  = Ux*[x-1,t+1]*Ut*[x-1,t]*Ux[x-1,t]
    Astaple_t = (Ux_tp1 * jnp.conj(Ut_xp1) * jnp.conj(Ux)
                    + jnp.conj(Ux_xm1_tp1) * jnp.conj(Ut_xm1) * Ux_xm1)
    guageTimeForce = settings.beta * jnp.imag(Ut * Astaple_t)

    Ut_tm1     = jnp.roll(Ut, shift=1,  axis=1)              # Ut[x, t-1]
    Ut_xp1_tm1 = jnp.roll(Ut_xp1, shift=1, axis=1)          # Ut[x+1, t-1]
    Ux_tm1     = jnp.roll(Ux, shift=1,  axis=1)              # Ux[x, t-1]

    # Space links: top staple    = Ut[x+1,t]*Ux*[x,t+1]*Ut*[x,t]
    #              bottom staple  = Ut*[x+1,t-1]*Ux*[x,t-1]*Ut[x,t-1]
    Astaple_x = (Ut_xp1 * jnp.conj(Ux_tp1) * jnp.conj(Ut)
                    + jnp.conj(Ut_xp1_tm1) * jnp.conj(Ux_tm1) * Ut_tm1)
    gaugeSpaceForce = settings.beta * jnp.imag(Ux * Astaple_x)

    # --- Fermion force: -2*Re<X| dD |Y>, minus sign from differentiating the inverse ---
    Force = jnp.stack([guageTimeForce, gaugeSpaceForce], axis=-1)
    
    Force -= fermionForceBilinear(settings, gaugeLinks, X, Y)
    Force += pvForce(settings, gaugeLinks, chi)

    return Force.astype(jnp.float64), X

@partial(jax.jit, static_argnums=0)
def pvAction(modelSettings: dwfParams, chi_pv, gaugeLinks):

    pvSettings = modelSettings._replace(fMass=1.0)

    v = applyD_dwf(pvSettings, gaugeLinks, chi_pv)

    return jnp.vdot(v, v).real

@partial(jax.jit, static_argnums=0)
def totalAction(modelSettings: dwfParams, gaugeLinks):
    # calculate all wilson loops (plaquettes)
    Ut = gaugeLinks[:,:,0] # Time links (shape: dimx, dimt)
    Ux = gaugeLinks[:,:,1] # Space links (shape: dimx, dimt)
    
    # Shift arrays to get U_t(x+1, t) and U_x(x, t+1)
    Ut_shifted_x = jnp.roll(Ut, shift=-1, axis=0) 
    Ux_shifted_t = jnp.roll(Ux, shift=-1, axis=1) 
    
    # Multiply the four sides of the plaquette
    # U_x(x,t) * U_t(x+1,t) * U_x*(x,t+1) * U_t*(x,t)
    plaq = Ux * Ut_shifted_x * jnp.conjugate(Ux_shifted_t) * jnp.conjugate(Ut)
    
    # Standard Wilson gauge action: S = beta * sum(1 - Re(U_plaq))
    action = modelSettings.beta * jnp.sum(1.0 - jnp.real(plaq))
    
    return action

@partial(jax.jit, static_argnums=(0, 3))
def hmcStep(settings:dwfParams, gaugeLinks, rngKey,  numSubSteps=100, cgRtolForce=1e-5,cgRtolAction=1e-10):
    kChi, kEta, kMom, kMetro = jax.random.split(rngKey, 4)

    shape = (settings.dim5, settings.dimx, settings.dimt, 2)
    chi = jax.random.normal(kChi, shape, dtype=jnp.complex128)
    eta = jax.random.normal(kEta, shape, dtype=jnp.complex128)
    conjPInitial = jax.random.normal(kMom, (settings.dimx, settings.dimt, 2))
    r = jax.random.uniform(kMetro)

    #initial fermion action is just \chi.\chi
    initialFermionAction = jnp.vdot(chi,chi).real

    #copy current gauge configuration
    gaugeLinksCopy = jnp.copy(gaugeLinks)

    epsilon=1/numSubSteps

    #generate pseduofermions field:
    phi = applyD_dwf(settings,gaugeLinks,chi)

    #generate pseduofermions field for pauli villars:
    pvSettings = settings._replace(fMass=1.0)
    DDdag_pv = lambda v: applyD_dwf(pvSettings, gaugeLinks, applyDdag_dwf(pvSettings, gaugeLinks, v))
    y, _ = cg(DDdag_pv, eta, tol=cgRtolAction)
    resPV = jnp.linalg.norm(DDdag_pv(y) - eta) / jnp.linalg.norm(eta)

    chi_pv = applyDdag_dwf(pvSettings, gaugeLinks, y)

    #first momentum half step:
    Force, X = hmcForcingFunction_vec(settings, gaugeLinksCopy,phi,chi_pv,cgRtol=cgRtolForce)
    conjP = conjPInitial - epsilon/2 * Force

    #full leapfrog steps as one compiled loop: carry = (links, momentum, CG warm start)
    def leapfrogBody(carry, _):
        U, P, Xprev = carry
        U = U * jnp.exp(1j*epsilon*P)
        F, Xnew = hmcForcingFunction_vec(settings, U, phi, chi_pv, x0=Xprev, cgRtol=cgRtolForce)
        return (U, P - epsilon*F, Xnew), None

    (gaugeLinksCopy, conjP, X), _ = jax.lax.scan(leapfrogBody,
                                                 (gaugeLinksCopy, conjP, X),
                                                 None, length=numSubSteps-1)

    #last step
    gaugeLinksCopy *= jnp.exp(1j*epsilon *conjP)
    Force, X = hmcForcingFunction_vec(settings,gaugeLinksCopy,phi,chi_pv, x0=X,cgRtol=cgRtolForce)
    conjP = conjP - epsilon/2 * Force

    finalFermionAction, resFinal = pseudoBilinear(settings, phi,gaugeLinksCopy,cgRtolAction)

    metroFactor = jnp.exp(0.5*jnp.sum(conjPInitial**2)-0.5*jnp.sum(conjP**2)
                            +totalAction(settings, gaugeLinks)-totalAction(settings, gaugeLinksCopy)
                            +initialFermionAction-finalFermionAction
                            +pvAction(settings, chi_pv,gaugeLinks)-pvAction(settings,chi_pv,gaugeLinksCopy))

    success = r<metroFactor

    gaugeLinksOut = jnp.where(success,
                           gaugeLinksCopy/jnp.abs(gaugeLinksCopy),
                           gaugeLinks)

    #return worst error so that convergence failure can be caught
    worstRes = jnp.maximum(resPV, resFinal)

    #metroFactor = exp(-dH), useful for the <exp(-dH)>=1 consistency check
    return gaugeLinksOut, success, metroFactor, worstRes


def hmcChain(modelSettings:dwfParams, metroSteps=1000, numSubSteps = 10,
              cgRtolForce=1e-5, cgRtolAction=1e-10,
              seed=0, tqdmPosition=0):

    chainKey = jax.random.key(seed)

    #host-side archive, filled per step (also pulls each config off the device)
    linkHistory = np.full((metroSteps, modelSettings.dimx,modelSettings.dimt,2),1+0j)
    #explicit dtype: a weakly-typed initial array has a different jit signature
    #than the strongly-typed one hmcStep returns, forcing a second compile
    gaugeLinks = jnp.full((modelSettings.dimx,modelSettings.dimt,2),1+0j,dtype=jnp.complex128)

    #per-step metropolis accept/reject record, filled by hmcChain
    acceptHistory = np.zeros(metroSteps, dtype=bool)
        
    for currentStep in tqdm(range(metroSteps), position=tqdmPosition, leave=True):
        stepKey = jax.random.fold_in(chainKey, currentStep)

        gaugeLinks, acceptHistory[currentStep], _, worstRes = hmcStep(modelSettings, gaugeLinks,stepKey,
                                                          numSubSteps=numSubSteps,
                                                          cgRtolForce=cgRtolForce,cgRtolAction=cgRtolAction)

        if float(worstRes) > 10 * cgRtolAction:
            raise RuntimeError(f"action/heat-bath CG failed: residual {float(worstRes):.3e}")

        linkHistory[currentStep] = gaugeLinks
    
    return modelSettings, linkHistory, acceptHistory


#chains live on axis 0 of gaugeLinks and rngKeys; settings/substeps/tolerances are
#shared. in_axes pairs with positional args only, so call this positionally:
#hmcStepBatch(settings, gaugeLinks, chainKeys, numSubSteps, cgRtolForce, cgRtolAction)
hmcStepBatch = partial(jax.jit, static_argnums=(0, 3))(
    jax.vmap(hmcStep, in_axes=(None, 0, 0, None, None, None)))


def hmcChainBatch(modelSettings: dwfParams, nChains, metroSteps=1000, numSubSteps=10,
                  cgRtolForce=1e-5, cgRtolAction=1e-10, seed=0, tqdmPosition=0):
    """Runs nChains chains in lockstep on the device via hmcStepBatch.

    Chain c's randomness at step n is deterministic in (seed, n, c), so a single
    seed reproduces the whole ensemble. Returns host arrays
      linkHistory   : (metroSteps, nChains, dimx, dimt, 2) complex
      acceptHistory : (metroSteps, nChains) bool
    """
    masterKey = jax.random.key(seed)

    gaugeLinks = jnp.full((nChains, modelSettings.dimx, modelSettings.dimt, 2),
                          1+0j, dtype=jnp.complex128)

    linkHistory = np.zeros((metroSteps, nChains, modelSettings.dimx, modelSettings.dimt, 2),
                           dtype=complex)
    acceptHistory = np.zeros((metroSteps, nChains), dtype=bool)

    for currentStep in tqdm(range(metroSteps), position=tqdmPosition, leave=True):
        chainKeys = jax.random.split(jax.random.fold_in(masterKey, currentStep), nChains)

        gaugeLinks, accept, _, worstRes = hmcStepBatch(modelSettings, gaugeLinks, chainKeys,
                                                       numSubSteps, cgRtolForce, cgRtolAction)

        worst = float(jnp.max(worstRes))
        if worst > 10 * cgRtolAction:
            raise RuntimeError(f"action/heat-bath CG failed: residual {worst:.3e}")

        acceptHistory[currentStep] = np.asarray(accept)
        linkHistory[currentStep] = np.asarray(gaugeLinks)

    return linkHistory, acceptHistory


def _applyD_core(settings: dwfParams, gaugeLinks, psi, sign):
    """D (sign=+1) or D^dagger (sign=-1): fused CUDA kernel when built, XLA otherwise."""
    if _USE_CUDA_KERNEL:
        return _dwf_cuda.applyD(settings, gaugeLinks, psi, sign)
    return _applyD_xla(settings, gaugeLinks, psi, sign)


def _applyD_xla(settings: dwfParams, gaugeLinks, psi, sign):
    """Shared body for D (sign=+1) and D^dagger (sign=-1).

    The gammas are Hermitian and the link placement is symmetric under the
    conjugate transpose, so D^dagger is D with every gamma negated: the
    forward/backward spin factors (1 -+ gamma) swap, and P- <-> P+ in s5.
    """

    #typing so that mixed precision works
    dataType = psi.dtype
    realType = jnp.zeros((), dataType).real.dtype          # f32 for c64, f64 for c128
    GX, GT, G5 = _GAMMA_X.astype(dataType), _GAMMA_T.astype(dataType), _GAMMA_5.astype(dataType)

    a = settings.a
    Ut = gaugeLinks[:, :, 0][None, :, :, None]  # (1, dimx, dimt, 1)
    Ux = gaugeLinks[:, :, 1][None, :, :, None]

    I2 = jnp.eye(2, dtype=psi.dtype)
    P_fwd5 = (I2 - sign * G5) / 2   # P- for D, P+ for D^dagger
    P_bwd5 = (I2 + sign * G5) / 2

    #diagonal: Wilson (−M5 + 2/a) plus the identity from D5
    out = (1 - settings.M5 + 2 / a) * psi

    #anti-periodic boundary in time: sign flip where roll wraps around
    bc_fwd = jnp.ones(settings.dimt,dtype=realType).at[-1].set(-1.0)[None, None, :, None]
    bc_bwd = jnp.ones(settings.dimt,dtype=realType).at[0].set(-1.0)[None, None, :, None]

    # +x hop: -1/2a * Ux(x,t) (1∓γx) ψ(x+1,t)
    out -= (1 / (2 * a)) * Ux * jnp.einsum('ij,sxtj->sxti', I2 - sign * GX,
                                           jnp.roll(psi, -1, axis=1))
    # -x hop: -1/2a * Ux*(x-1,t) (1±γx) ψ(x-1,t)
    out -= (1 / (2 * a)) * jnp.roll(jnp.conj(Ux), 1, axis=1) \
           * jnp.einsum('ij,sxtj->sxti', I2 + sign * GX, jnp.roll(psi, 1, axis=1))
    # +t hop: -1/2a * Ut(x,t) (1∓γt) ψ(x,t+1)
    out -= (1 / (2 * a)) * Ut * jnp.einsum('ij,sxtj->sxti', I2 - sign * GT,
                                           bc_fwd * jnp.roll(psi, -1, axis=2))
    # -t hop: -1/2a * Ut*(x,t-1) (1±γt) ψ(x,t-1)
    out -= (1 / (2 * a)) * jnp.roll(jnp.conj(Ut), 1, axis=2) \
           * jnp.einsum('ij,sxtj->sxti', I2 + sign * GT, bc_bwd * jnp.roll(psi, 1, axis=2))

    #5th-dimension hops: the roll wrap-around carries a factor of -fMass so the
    #mass terms +m P appear on the boundaries
    m_fwd = jnp.ones(settings.dim5,dtype=realType).at[-1].set(-settings.fMass)[:, None, None, None]
    m_bwd = jnp.ones(settings.dim5,dtype=realType).at[0].set(-settings.fMass)[:, None, None, None]

    out -= m_fwd * jnp.einsum('ij,sxtj->sxti', P_fwd5, jnp.roll(psi, -1, axis=0))
    out -= m_bwd * jnp.einsum('ij,sxtj->sxti', P_bwd5, jnp.roll(psi, 1, axis=0))

    return out


def applyD_dwf(settings: dwfParams, gaugeLinks, psi):
    """Matrix-free D_dwf, equivalent to buildDwfOp(settings, gaugeLinks) @ psi.flatten().

    psi: (dim5, dimx, dimt, 2). Returns the same shape.
    The Wilson term (mass -M5) acts identically on every s5 slice; the 5th-dim
    hops couple neighboring slices through the chiral projectors, with the
    physical mass fMass entering only on the wrap-around between s=N5-1 and s=0.
    """
    return _applyD_core(settings, gaugeLinks, psi, +1)


def applyDdag_dwf(settings: dwfParams, gaugeLinks, psi):
    """Matrix-free D_dwf^dagger, equivalent to buildDwfOp(...).conj().T @ psi.flatten()."""
    return _applyD_core(settings, gaugeLinks, psi, -1)





