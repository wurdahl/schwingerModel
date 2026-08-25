import numpy as np
from scipy.sparse.linalg import cg

from tqdm import tqdm

from .params import LatticeParams, dwfParams
from .buildOps import buildDwfOp
from .hmc import stapleCalc, totalAction

def pseudoBilinear(modelSettings: dwfParams, pseudoField, gaugeLinks, cgRtol):
    diracOp = buildDwfOp(modelSettings,gaugeLinks)
    dDag = diracOp.conj().T

    D_pv = buildDwfOp(modelSettings._replace(fMass=1.0), gaugeLinks)

    regulatedField = D_pv@pseudoField

    X, exitcode = cg(diracOp @ dDag, regulatedField, rtol= cgRtol)

    if exitcode != 0:
        raise RuntimeError(f"Conjugate Gradient failed to converge! Exit code: {exitcode}")

    return np.vdot(regulatedField,X).real



def fermionForceBilinear(settings: dwfParams, gaugeLinks, left, right):
    """2*Re <left| dD/dtheta |right> at each link, summed over the s5 slices.

    The gauge links enter D_dwf only through the Wilson term, replicated on every
    s5 slice, so dD/dtheta is the same per-slice expression as the Wilson case
    (and the same for any fMass, so the Pauli-Villars force can reuse this).
    left, right: shape (dim5, dimx, dimt, 2). Returns real (dimx, dimt, 2).
    """
    I = np.eye(2, dtype=np.complex128)
    c = 1j / (2 * settings.a)

    Ux = gaugeLinks[:, :, 1]  # (dimx, dimt)
    Ut = gaugeLinks[:, :, 0]  # (dimx, dimt)

    P_minus_x = I - settings.gammax
    P_plus_x  = I + settings.gammax
    P_minus_t = I - settings.gammat
    P_plus_t  = I + settings.gammat

    #axis 0 is s5, axis 1 is x, axis 2 is t
    R_xp1 = np.roll(right, shift=-1, axis=1)  # right[s, x+1, t, :]
    L_xp1 = np.roll(left,  shift=-1, axis=1)  # left[s, x+1, t, :]
    R_tp1 = np.roll(right, shift=-1, axis=2)  # right[s, x, t+1, :]
    L_tp1 = np.roll(left,  shift=-1, axis=2)  # left[s, x, t+1, :]

    out = np.zeros((settings.dimx, settings.dimt, 2))

    # Spatial: Z_x = -c*Ux * <L|P-_x|R_{x+1}> + c*Ux* * <L_{x+1}|P+_x|R>
    Pm_x_R_xp1 = np.einsum('ij,sxyj->sxyi', P_minus_x, R_xp1,optimize=True)
    Pp_x_R     = np.einsum('ij,sxyj->sxyi', P_plus_x,  right,optimize=True)
    Z_x = (-c * Ux      * np.einsum('sxyi,sxyi->xy', np.conj(left),  Pm_x_R_xp1,optimize=True)
            + c * np.conj(Ux) * np.einsum('sxyi,sxyi->xy', np.conj(L_xp1), Pp_x_R,optimize=True))
    out[:, :, 1] = 2 * Z_x.real

    # Time: Z_t = -c*Ut * <L|P-_t|R_{t+1}> + c*Ut* * <L_{t+1}|P+_t|R>
    Pm_t_R_tp1 = np.einsum('ij,sxyj->sxyi', P_minus_t, R_tp1, optimize=True)
    Pp_t_R     = np.einsum('ij,sxyj->sxyi', P_plus_t,  right, optimize=True)
    Z_t = (-c * Ut      * np.einsum('sxyi,sxyi->xy', np.conj(left),  Pm_t_R_tp1, optimize=True)
            + c * np.conj(Ut) * np.einsum('sxyi,sxyi->xy', np.conj(L_tp1), Pp_t_R, optimize=True))

    # Anti-periodic boundary condition: flip sign at t = dimt-1
    bc_t = np.ones((settings.dimx, settings.dimt))
    bc_t[:, -1] = -1.0
    out[:, :, 0] = 2 * Z_t.real * bc_t

    return out


def hmcForcingFunction_vec(settings: dwfParams, gaugeLinks, phis, x0=None, cgRtol=1e-5):
    #force lives on the 2D gauge links: the s5 axis exists only on the fermion fields
    Force = np.zeros((settings.dimx, settings.dimt, 2))

    # --- CG solve (same as original) ---
    diracOp = buildDwfOp(settings, gaugeLinks)
    dDag = diracOp.conj().T

    #X is (D D^\dagger)^{-1}D(1)\phi

    D_pv = buildDwfOp(settings._replace(fMass=1.0), gaugeLinks)

    X, exitcode = cg(diracOp@dDag, D_pv@phis, x0=x0, rtol=cgRtol)

    if exitcode != 0:
        raise RuntimeError(f"Conjugate Gradient failed to converge! Exit code: {exitcode}")

    #Y = D^\dagger X
    Y = (dDag)@X

    Y = np.reshape(Y,(settings.dim5, settings.dimx, settings.dimt, 2))

    Xflat = X
    X = np.reshape(X,(settings.dim5, settings.dimx, settings.dimt, 2))

    Ux = gaugeLinks[:, :, 1]  # (dimx, dimt)
    Ut = gaugeLinks[:, :, 0]  # (dimx, dimt)

    # --- Gauge force (vectorized staples) ---
    Ux_tp1     = np.roll(Ux, shift=-1, axis=1)              # Ux[x, t+1]
    Ut_xp1     = np.roll(Ut, shift=-1, axis=0)              # Ut[x+1, t]
    Ux_xm1     = np.roll(Ux, shift=1,  axis=0)              # Ux[x-1, t]
    Ux_xm1_tp1 = np.roll(Ux_tp1, shift=1, axis=0)          # Ux[x-1, t+1]
    Ut_xm1     = np.roll(Ut, shift=1,  axis=0)              # Ut[x-1, t]

    # Time links: right staple = Ux[x,t+1]*Ut*[x+1,t]*Ux*[x,t]
    #             left staple  = Ux*[x-1,t+1]*Ut*[x-1,t]*Ux[x-1,t]
    Astaple_t = (Ux_tp1 * np.conj(Ut_xp1) * np.conj(Ux)
                    + np.conj(Ux_xm1_tp1) * np.conj(Ut_xm1) * Ux_xm1)
    Force[:, :, 0] = settings.beta * np.imag(Ut * Astaple_t)

    Ut_tm1     = np.roll(Ut, shift=1,  axis=1)              # Ut[x, t-1]
    Ut_xp1_tm1 = np.roll(Ut_xp1, shift=1, axis=1)          # Ut[x+1, t-1]
    Ux_tm1     = np.roll(Ux, shift=1,  axis=1)              # Ux[x, t-1]

    # Space links: top staple    = Ut[x+1,t]*Ux*[x,t+1]*Ut*[x,t]
    #              bottom staple  = Ut*[x+1,t-1]*Ux*[x,t-1]*Ut[x,t-1]
    Astaple_x = (Ut_xp1 * np.conj(Ux_tp1) * np.conj(Ut)
                    + np.conj(Ut_xp1_tm1) * np.conj(Ux_tm1) * Ut_tm1)
    Force[:, :, 1] = settings.beta * np.imag(Ux * Astaple_x)

    # Fermion force: 2*Re<X| dD |phis-Y> 
    phi4 = np.reshape(phis, (settings.dim5, settings.dimx, settings.dimt, 2))
    Force += fermionForceBilinear(settings, gaugeLinks, X, phi4 - Y)

    return Force, Xflat


def hmcStep(modelSettings:dwfParams, gaugeLinks, numSubSteps=100, rng=None,cgRtolForce=1e-5,cgRtolAction=1e-10,
            coldStartForce=False):
    #a shared default generator would correlate independent callers, so make a fresh one
    if rng is None:
        rng = np.random.default_rng()

    #copy current gauge configuration
    gaugeLinksCopy = np.copy(gaugeLinks)

    epsilon=1/numSubSteps

    #generate pseduofermions field:
    chi = (rng.normal(loc=0,scale=1/np.sqrt(2),size=(modelSettings.dim5*modelSettings.dimx*modelSettings.dimt*2))
            +1j*rng.normal(loc=0,scale=1/np.sqrt(2),size=(modelSettings.dim5*modelSettings.dimx*modelSettings.dimt*2)))

    #initial fermion action is just \chi.\chi
    initialFermionAction = np.vdot(chi,chi).real    

    pvSettings = modelSettings._replace(fMass=1)
    #Ratio heat-bath phi = D(1)^{-1} D(m) chi via D(1)^dag (D(1) D(1)^dag)^{-1} D(m) chi,
    #so that phi^dag D(1)^dag (D(m) D(m)^dag)^{-1} D(1) phi = chi^dag chi. The fMass=1
    #operator is well-conditioned, so CG beats a direct sparse LU by ~100x.
    #Tight rtol because heat-bath error is not corrected by the Metropolis step.
    D_pv = buildDwfOp(pvSettings, gaugeLinks)
    D_pv_dag = D_pv.conj().T
    y, exitcode = cg(D_pv @ D_pv_dag, buildDwfOp(modelSettings,gaugeLinks)@chi, rtol=1e-12)
    if exitcode != 0:
        raise RuntimeError(f"PV heat-bath CG failed to converge! Exit code: {exitcode}")
    phi = D_pv_dag @ y

    #generate initial value for conjugate field
    conjPInitial = rng.normal(loc=0,scale=1,size=(modelSettings.dimx,modelSettings.dimt,2))

    #first momentum half step:
    Force, X = hmcForcingFunction_vec(modelSettings, gaugeLinksCopy,phi,cgRtol=cgRtolForce)
    conjP = conjPInitial - epsilon/2 * Force
    #coldStartForce: restart every force CG from zero so the force is a
    #deterministic function of U and the leapfrog stays exactly reversible
    #(warm starts measurably violate <exp(-dH)>=1); costs ~40% more solve time
    for i in range(numSubSteps-1):
        gaugeLinksCopy *= np.exp(1j*epsilon *conjP)
        Force, X = hmcForcingFunction_vec(modelSettings,gaugeLinksCopy,phi,
                                          x0=None if coldStartForce else X,cgRtol=cgRtolForce)
        conjP = conjP - epsilon * Force
    #last step
    gaugeLinksCopy *= np.exp(1j*epsilon *conjP)
    Force, X = hmcForcingFunction_vec(modelSettings,gaugeLinksCopy,phi,
                                      x0=None if coldStartForce else X,cgRtol=cgRtolForce)
    conjP = conjP - epsilon/2 * Force

    metroFactor = np.exp(0.5*np.sum(conjPInitial**2)-0.5*np.sum(conjP**2)
                            +totalAction(modelSettings, gaugeLinks)-totalAction(modelSettings, gaugeLinksCopy)
                            +initialFermionAction-pseudoBilinear(modelSettings,phi,gaugeLinksCopy,cgRtolAction))
    r=rng.random()
    if(r<metroFactor):
        success=True
        gaugeLinks = gaugeLinksCopy
        gaugeLinks/= np.abs(gaugeLinks)
    else:
        success=False

    #metroFactor = exp(-dH), useful for the <exp(-dH)>=1 consistency check
    return gaugeLinks, success, metroFactor

def hmcChain(modelSettings:dwfParams, metroSteps=1000, numSubSteps = 10,
              cgRtolForce=1e-5, cgRtolAction=1e-10,
              seed=0, tqdmPosition=0, coldStartForce=False):

    rng = np.random.default_rng(seed)

    linkHistory = np.full((metroSteps, modelSettings.dimx,modelSettings.dimt,2),1+0j)
    gaugeLinks = np.full((modelSettings.dimx,modelSettings.dimt,2),1+0j)

    #per-step metropolis accept/reject record, filled by hmcChain
    acceptHistory = np.zeros(metroSteps, dtype=bool)
        
    for currentStep in tqdm(range(metroSteps), position=tqdmPosition, leave=True):

        gaugeLinks, acceptHistory[currentStep], _ = hmcStep(modelSettings, gaugeLinks,
                                                          numSubSteps=numSubSteps, rng=rng,
                                                          cgRtolForce=cgRtolForce,cgRtolAction=cgRtolAction,
                                                          coldStartForce=coldStartForce)


        linkHistory[currentStep] = gaugeLinks
    
    return modelSettings, linkHistory, acceptHistory
