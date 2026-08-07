from functools import partial

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.scipy.sparse.linalg import cg

import numpy as np
from tqdm import tqdm

from .params import dwfParams

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
    I = jnp.eye(2, dtype=jnp.complex128)
    c = 1j / (2 * settings.a)

    Ux = gaugeLinks[:, :, 1]  # (dimx, dimt)
    Ut = gaugeLinks[:, :, 0]  # (dimx, dimt)

    P_minus_x = I - _GAMMA_X
    P_plus_x  = I + _GAMMA_X
    P_minus_t = I - _GAMMA_T
    P_plus_t  = I + _GAMMA_T

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
    bc_t = jnp.ones((settings.dimx, settings.dimt)).at[:, -1].set(-1.0)
    
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

    operator = lambda v: applyD_dwf(settings, gaugeLinks, applyDdag_dwf(settings, gaugeLinks, v))
    
    X, _ = cg(operator, phis, tol=cgRtol,x0=x0)
    relRes = jnp.linalg.norm(operator(X) - phis) / jnp.linalg.norm(phis)

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

    return Force, X

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

def hmcStep(settings:dwfParams, gaugeLinks, rngKey,  numSubSteps=100, cgRtolForce=1e-5,cgRtolAction=1e-10):
    kChi, kEta, kMom, kMetro = jax.random.split(rngKey, 4)

    shape = (settings.dim5, settings.dimx, settings.dimt, 2)
    chi = jax.random.normal(kChi, shape, dtype=jnp.complex128)
    eta = jax.random.normal(kEta, shape, dtype=jnp.complex128)
    conjPInitial = jax.random.normal(kMom, (settings.dimx, settings.dimt, 2))
    r = jax.random.uniform(kMetro)

    #copy current gauge configuration
    gaugeLinksCopy = jnp.copy(gaugeLinks)

    epsilon=1/numSubSteps

    #generate pseduofermions field:
    phi = applyD_dwf(settings,gaugeLinks,chi)

    #generate pseduofermions field for pauli villars:
    pvSettings = settings._replace(fMass=1.0)
    DDdag_pv = lambda v: applyD_dwf(pvSettings, gaugeLinks, applyDdag_dwf(pvSettings, gaugeLinks, v))
    y, _ = cg(DDdag_pv, eta, tol=cgRtolAction)
    chi_pv = applyDdag_dwf(pvSettings, gaugeLinks, y)


    #first momentum half step:
    Force, X = hmcForcingFunction_vec(settings, gaugeLinksCopy,phi,chi_pv,cgRtol=cgRtolForce)
    conjP = conjPInitial - epsilon/2 * Force
    for i in range(numSubSteps-1):
        gaugeLinksCopy *= jnp.exp(1j*epsilon *conjP)
        Force, X = hmcForcingFunction_vec(settings,gaugeLinksCopy,phi,chi_pv,x0=X,cgRtol=cgRtolForce)
        conjP = conjP - epsilon * Force
    #last step
    gaugeLinksCopy *= jnp.exp(1j*epsilon *conjP)
    Force, X = hmcForcingFunction_vec(settings,gaugeLinksCopy,phi,chi_pv, x0=X,cgRtol=cgRtolForce)
    conjP = conjP - epsilon/2 * Force

    initialFermionAction, resInitial = pseudoBilinear(settings, phi,gaugeLinks,cgRtolAction)
    finalFermionAction, resFinal = pseudoBilinear(settings, phi,gaugeLinksCopy,cgRtolAction)

    #the action solves enter dH directly, so an unconverged solve biases the
    #accept/reject silently (jax cg reports nothing at maxiter): fail loudly.
    #10x slack because jax's stopping criterion can land right at the boundary.
    worstRes = float(jnp.maximum(resInitial, resFinal))
    if worstRes > 10*cgRtolAction:
        raise RuntimeError(f"Action CG failed to converge! Relative residual: {worstRes:.3e}")

    metroFactor = jnp.exp(0.5*jnp.sum(conjPInitial**2)-0.5*jnp.sum(conjP**2)
                            +totalAction(settings, gaugeLinks)-totalAction(settings, gaugeLinksCopy)
                            +initialFermionAction-finalFermionAction
                            +pvAction(settings, chi_pv,gaugeLinks)-pvAction(settings,chi_pv,gaugeLinksCopy))

    if(r<metroFactor):
        success=True
        gaugeLinks = gaugeLinksCopy
        gaugeLinks/= jnp.abs(gaugeLinks)
    else:
        success=False

    #metroFactor = exp(-dH), useful for the <exp(-dH)>=1 consistency check
    return gaugeLinks, success, metroFactor


def hmcChain(modelSettings:dwfParams, metroSteps=1000, numSubSteps = 10,
              cgRtolForce=1e-5, cgRtolAction=1e-10,
              seed=0, tqdmPosition=0):

    chainKey = jax.random.key(seed)

    linkHistory = np.full((metroSteps, modelSettings.dimx,modelSettings.dimt,2),1+0j)
    gaugeLinks = jnp.full((modelSettings.dimx,modelSettings.dimt,2),1+0j)

    #per-step metropolis accept/reject record, filled by hmcChain
    acceptHistory = np.zeros(metroSteps, dtype=bool)
        
    for currentStep in tqdm(range(metroSteps), position=tqdmPosition, leave=True):
        stepKey = jax.random.fold_in(chainKey, currentStep)

        gaugeLinks, acceptHistory[currentStep], _ = hmcStep(modelSettings, gaugeLinks,stepKey,
                                                          numSubSteps=numSubSteps,
                                                          cgRtolForce=cgRtolForce,cgRtolAction=cgRtolAction)


        linkHistory[currentStep] = gaugeLinks
    
    return modelSettings, linkHistory, acceptHistory



def _applyD_core(settings: dwfParams, gaugeLinks, psi, sign):
    """Shared body for D (sign=+1) and D^dagger (sign=-1).

    The gammas are Hermitian and the link placement is symmetric under the
    conjugate transpose, so D^dagger is D with every gamma negated: the
    forward/backward spin factors (1 -+ gamma) swap, and P- <-> P+ in s5.
    """
    a = settings.a
    Ut = gaugeLinks[:, :, 0][None, :, :, None]  # (1, dimx, dimt, 1)
    Ux = gaugeLinks[:, :, 1][None, :, :, None]

    I2 = jnp.eye(2, dtype=psi.dtype)
    P_fwd5 = (I2 - sign * _GAMMA_5) / 2   # P- for D, P+ for D^dagger
    P_bwd5 = (I2 + sign * _GAMMA_5) / 2

    #diagonal: Wilson (−M5 + 2/a) plus the identity from D5
    out = (1 - settings.M5 + 2 / a) * psi

    #anti-periodic boundary in time: sign flip where roll wraps around
    bc_fwd = jnp.ones(settings.dimt).at[-1].set(-1.0)[None, None, :, None]
    bc_bwd = jnp.ones(settings.dimt).at[0].set(-1.0)[None, None, :, None]

    # +x hop: -1/2a * Ux(x,t) (1∓γx) ψ(x+1,t)
    out -= (1 / (2 * a)) * Ux * jnp.einsum('ij,sxtj->sxti', I2 - sign * _GAMMA_X,
                                           jnp.roll(psi, -1, axis=1))
    # -x hop: -1/2a * Ux*(x-1,t) (1±γx) ψ(x-1,t)
    out -= (1 / (2 * a)) * jnp.roll(jnp.conj(Ux), 1, axis=1) \
           * jnp.einsum('ij,sxtj->sxti', I2 + sign * _GAMMA_X, jnp.roll(psi, 1, axis=1))
    # +t hop: -1/2a * Ut(x,t) (1∓γt) ψ(x,t+1)
    out -= (1 / (2 * a)) * Ut * jnp.einsum('ij,sxtj->sxti', I2 - sign * _GAMMA_T,
                                           bc_fwd * jnp.roll(psi, -1, axis=2))
    # -t hop: -1/2a * Ut*(x,t-1) (1±γt) ψ(x,t-1)
    out -= (1 / (2 * a)) * jnp.roll(jnp.conj(Ut), 1, axis=2) \
           * jnp.einsum('ij,sxtj->sxti', I2 + sign * _GAMMA_T, bc_bwd * jnp.roll(psi, 1, axis=2))

    #5th-dimension hops: the roll wrap-around carries a factor of -fMass so the
    #mass terms +m P appear on the boundaries
    m_fwd = jnp.ones(settings.dim5).at[-1].set(-settings.fMass)[:, None, None, None]
    m_bwd = jnp.ones(settings.dim5).at[0].set(-settings.fMass)[:, None, None, None]

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





