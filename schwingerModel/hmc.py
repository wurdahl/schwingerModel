import numpy as np
from scipy.sparse.linalg import cg
from scipy.sparse.linalg import splu

from tqdm import tqdm

from .params import LatticeParams
from .buildOps import buildDiracOp
from .topology import instanton

def logAbsDetD(D):                      # D: sparse Dirac operator
    lu = splu(D.tocsc(), permc_spec="MMD_AT_PLUS_A")
    return np.log(np.abs(lu.U.diagonal())).sum()

def pseudoBilinear(modelSettings: LatticeParams, pseudoField, gaugeLinks, cgRtol):
    diracOp = buildDiracOp(modelSettings,gaugeLinks)
    dDag = diracOp.conj().T

    X, exitcode = cg(diracOp @ dDag, pseudoField, rtol= cgRtol)

    if exitcode != 0:
        raise RuntimeError(f"Conjugate Gradient failed to converge! Exit code: {exitcode}")

    return np.vdot(pseudoField,X).real

def stapleCalc(modelSettings: LatticeParams,xIndex,tIndex,directionIndex, gaugeLinks):
    x, t, d = xIndex, tIndex, directionIndex
    
    # If link is in the TIME direction (U_t at x,t)
    if d == 0: 
        # Right staple (+x direction loop): U_x(x,t+1) * U_t*(x+1,t) * U_x*(x,t)
        staple_right = (gaugeLinks[x, (t+1)%modelSettings.dimt,1] * np.conjugate(gaugeLinks[(x+1)%modelSettings.dimx, t,0]) * np.conjugate(gaugeLinks[x, t,1]))
        
        # Left staple (-x direction loop): U_x*(x-1,t+1) * U_t*(x-1,t) * U_x(x-1,t)
        # (Note: Your original code actually had this specific staple mostly right!)
        staple_left = (np.conjugate(gaugeLinks[(x-1)%modelSettings.dimx, (t+1)%modelSettings.dimt,1]) * np.conjugate(gaugeLinks[(x-1)%modelSettings.dimx, t,0]) * gaugeLinks[(x-1)%modelSettings.dimx, t,1])
        
        Astaple = staple_right + staple_left

    # If link is in the SPACE direction (U_x at x,t)
    if d == 1: 
        # Top staple (+t direction loop): U_t(x+1,t) * U_x*(x,t+1) * U_t*(x,t)
        staple_top = (gaugeLinks[(x+1)%modelSettings.dimx, t,0] * np.conjugate(gaugeLinks[x, (t+1)%modelSettings.dimt,1]) * np.conjugate(gaugeLinks[x, t, 0]))
        
        # Bottom staple (-t direction loop): U_t*(x+1,t-1) * U_x*(x,t-1) * U_t(x,t-1)
        staple_bottom = (np.conjugate(gaugeLinks[(x+1)%modelSettings.dimx, (t-1)%modelSettings.dimt,0]) * np.conjugate(gaugeLinks[x, (t-1)%modelSettings.dimt,1]) * gaugeLinks[x, (t-1)%modelSettings.dimt,0])
        
        Astaple = staple_top + staple_bottom
    
    return Astaple

def totalAction(modelSettings: LatticeParams, gaugeLinks):
        # calculate all wilson loops (plaquettes)
        Ut = gaugeLinks[:,:,0] # Time links (shape: dimx, dimt)
        Ux = gaugeLinks[:,:,1] # Space links (shape: dimx, dimt)
        
        # Shift arrays to get U_t(x+1, t) and U_x(x, t+1)
        Ut_shifted_x = np.roll(Ut, shift=-1, axis=0) 
        Ux_shifted_t = np.roll(Ux, shift=-1, axis=1) 
        
        # Multiply the four sides of the plaquette
        # U_x(x,t) * U_t(x+1,t) * U_x*(x,t+1) * U_t*(x,t)
        plaq = Ux * Ut_shifted_x * np.conjugate(Ux_shifted_t) * np.conjugate(Ut)
        
        # Standard Wilson gauge action: S = beta * sum(1 - Re(U_plaq))
        action = modelSettings.beta * np.sum(1.0 - np.real(plaq))
        
        return action

#calculates the derivative of the action with respect to the generalized coordiantes Q (i.e. the fields)
#this describes how the conjugate momentum at each link will change during integration
def hmcForcingFunction(modelSettings: LatticeParams,gaugeLinks,phis, x0=None, cgRtol=1e-5):
    #Force is in the algebra of U(1), so it is always real
    Force = np.zeros((modelSettings.dimx,modelSettings.dimt, 2))
    #gauge field contribution

    #calculate cg for fermion force calculation
    diracOp = buildDiracOp(modelSettings, gaugeLinks)
    dDag = diracOp.conj().T

    #X is (D D^\dagger)^{-1}\phi
    X, exitcode = cg(diracOp@dDag, phis, x0=x0, rtol=cgRtol)

    if exitcode != 0:
        raise RuntimeError(f"Conjugate Gradient failed to converge! Exit code: {exitcode}")
    
    #Y = D^\dagger X
    Y = (dDag)@X

    Y = np.reshape(Y,(modelSettings.dimx,modelSettings.dimt,2))

    Xflat = X
    X = np.reshape(X,(modelSettings.dimx,modelSettings.dimt,2))

    #identity matrix and constant
    I = np.eye(2, dtype=np.complex128)
    c = 1j / (2 * modelSettings.a)

    for x in range(modelSettings.dimx):
        for t in range(modelSettings.dimt):
            for d in range(2):
                Astaple = stapleCalc(modelSettings, x,t,d,gaugeLinks)
                Force[x,t,d] = modelSettings.beta * np.imag(gaugeLinks[x,t,d]*Astaple)

            #fermion component of the force

            xp1 = (x + 1) % modelSettings.dimx
            tp1 = (t + 1) % modelSettings.dimt

            #spatial direction
            Z = (np.vdot(X[x,t], -c*(I-modelSettings.gammax)*gaugeLinks[x,t,1] @ Y[xp1,t])
                            + np.vdot(X[xp1,t], c*(I+modelSettings.gammax)*np.conjugate(gaugeLinks[x,t,1]) @ Y[x,t]))
            Force[x,t,1] -=2*Z.real

            #time direction
            Z = (np.vdot(X[x,t], -c*(I-modelSettings.gammat)*gaugeLinks[x,t,0] @ Y[x,tp1])
                            + np.vdot(X[x,tp1],c*(I+modelSettings.gammat)*np.conjugate(gaugeLinks[x,t,0]) @ Y[x,t]))
            
            #enforce antiperiodic boundary condition
            bc_t = -1.0 if t == modelSettings.dimt - 1 else 1.0

            Force[x,t,0] -= 2*Z.real * bc_t

    return Force, Xflat
    

def hmcForcingFunction_vec(settings: LatticeParams, gaugeLinks, phis, x0=None, cgRtol=1e-5):
    Force = np.zeros((settings.dimx, settings.dimt, 2))

    # --- CG solve (same as original) ---
    diracOp = buildDiracOp(settings, gaugeLinks)
    dDag = diracOp.conj().T

    #X is (D D^\dagger)\phi
    X, exitcode = cg(diracOp@dDag, phis, x0=x0, rtol=cgRtol)

    if exitcode != 0:
        raise RuntimeError(f"Conjugate Gradient failed to converge! Exit code: {exitcode}")
    
    #Y = D^\dagger X
    # Y = self.apply_D_vectorized(X, gaugeLinks,dagger=True)
    Y = (dDag)@X

    Y = np.reshape(Y,(settings.dimx,settings.dimt,2))

    Xflat = X
    X = np.reshape(X,(settings.dimx,settings.dimt,2))

    I = np.eye(2, dtype=np.complex128)
    c = 1j / (2 * settings.a)

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

    # --- Fermion force (vectorized inner products) ---
    P_minus_x = I - settings.gammax
    P_plus_x  = I + settings.gammax
    P_minus_t = I - settings.gammat
    P_plus_t  = I + settings.gammat

    Y_xp1 = np.roll(Y, shift=-1, axis=0)  # Y[x+1, t, :]
    X_xp1 = np.roll(X, shift=-1, axis=0)  # X[x+1, t, :]
    Y_tp1 = np.roll(Y, shift=-1, axis=1)  # Y[x, t+1, :]
    X_tp1 = np.roll(X, shift=-1, axis=1)  # X[x, t+1, :]

    # Spatial: Z_x = -c*Ux * <X|P-_x|Y_{x+1}> + c*Ux* * <X_{x+1}|P+_x|Y>
    Pm_x_Y_xp1 = np.einsum('ij,xyj->xyi', P_minus_x, Y_xp1)
    Pp_x_Y     = np.einsum('ij,xyj->xyi', P_plus_x,  Y)
    Z_x = (-c * Ux      * np.einsum('xyi,xyi->xy', np.conj(X),     Pm_x_Y_xp1)
            + c * np.conj(Ux) * np.einsum('xyi,xyi->xy', np.conj(X_xp1), Pp_x_Y))
    Force[:, :, 1] -= 2 * Z_x.real

    # Time: Z_t = -c*Ut * <X|P-_t|Y_{t+1}> + c*Ut* * <X_{t+1}|P+_t|Y>
    Pm_t_Y_tp1 = np.einsum('ij,xyj->xyi', P_minus_t, Y_tp1,optimize=True)
    Pp_t_Y     = np.einsum('ij,xyj->xyi', P_plus_t,  Y, optimize=True)
    Z_t = (-c * Ut      * np.einsum('xyi,xyi->xy', np.conj(X),     Pm_t_Y_tp1, optimize=True)
            + c * np.conj(Ut) * np.einsum('xyi,xyi->xy', np.conj(X_tp1), Pp_t_Y, optimize=True))

    # Anti-periodic boundary condition: flip sign at t = dimt-1
    bc_t = np.ones((settings.dimx, settings.dimt))
    bc_t[:, -1] = -1.0
    Force[:, :, 0] -= 2 * Z_t.real * bc_t

    return Force, Xflat


#do one step of an hmc metropolis algorithm
#returns boolean of success of total step
#if successful, replaces global value of gaugeLinks
def hmcStep(modelSettings:LatticeParams, gaugeLinks, numSubSteps=100, rng=None,cgRtol=1e-5):
    #a shared default generator would correlate independent callers, so make a fresh one
    if rng is None:
        rng = np.random.default_rng()

    #copy current gauge configuration
    gaugeLinksCopy = np.copy(gaugeLinks)

    epsilon=1/numSubSteps

    #generate pseduofermions field:
    chi = (rng.normal(loc=0,scale=1/np.sqrt(2),size=(modelSettings.dimx*modelSettings.dimt*2))
            +1j*rng.normal(loc=0,scale=1/np.sqrt(2),size=(modelSettings.dimx*modelSettings.dimt*2)))

    # phi = self.apply_D_vectorized(chi,self.gaugeLinks)
    phi = buildDiracOp(modelSettings,gaugeLinks)@chi

    #generate initial value for conjugate field
    conjPInitial = rng.normal(loc=0,scale=1,size=(modelSettings.dimx,modelSettings.dimt,2))

    #first momentum half step:
    Force, X = hmcForcingFunction_vec(modelSettings, gaugeLinksCopy,phi,cgRtol=cgRtol)
    conjP = conjPInitial - epsilon/2 * Force
    for i in range(numSubSteps-1):
        gaugeLinksCopy *= np.exp((1j)*epsilon *conjP)
        Force, X = hmcForcingFunction_vec(modelSettings,gaugeLinksCopy,phi,x0=X,cgRtol=cgRtol)
        conjP = conjP - epsilon * Force
    #last step
    gaugeLinksCopy *= np.exp((1j)*epsilon *conjP)
    Force, X = hmcForcingFunction_vec(modelSettings,gaugeLinksCopy,phi,x0=X,cgRtol=cgRtol)
    conjP = conjP - epsilon/2 * Force

    metroFactor = np.exp(0.5*np.sum(conjPInitial**2)-0.5*np.sum(conjP**2)
                            +totalAction(modelSettings, gaugeLinks)-totalAction(modelSettings, gaugeLinksCopy)
                            +pseudoBilinear(modelSettings, phi,gaugeLinks,cgRtol)-pseudoBilinear(modelSettings,phi,gaugeLinksCopy,cgRtol))
    r=rng.random()
    if(r<metroFactor):
        success=True
        gaugeLinks = gaugeLinksCopy
        gaugeLinks/= np.abs(gaugeLinks)
    else:
        success=False

    return gaugeLinks, success

def tunnelStep(modelSettings:LatticeParams, gaugeLinks,rng=None):
    if rng is None:
        rng = np.random.default_rng()

    gaugeLinksCopy = np.copy(gaugeLinks)

    dQ = rng.choice([-1,1])

    instantonField = instanton(dQ, modelSettings.dimx,modelSettings.dimt)

    gaugeLinksCopy*=instantonField

    diracCurrent = buildDiracOp(modelSettings, gaugeLinks)
    diracProp = buildDiracOp(modelSettings, gaugeLinksCopy)

    metroFactor = np.exp(totalAction(modelSettings,gaugeLinks)-totalAction(modelSettings,gaugeLinksCopy)
                            -2*logAbsDetD(diracCurrent)+2*logAbsDetD(diracProp))

    r=rng.random()
    if(r<metroFactor):
        success=True
        gaugeLinks = gaugeLinksCopy
    else:
        success=False

    return gaugeLinks, success

def hmcChain(modelSettings:LatticeParams, metroSteps=1000, numSubSteps = 10, cgRtol=1e-5,
             tunneling=True, seed=0, tqdmPosition=0):

    rng = np.random.default_rng(seed)

    linkHistory = np.full((metroSteps, modelSettings.dimx,modelSettings.dimt,2),1+0j)
    gaugeLinks = np.full((modelSettings.dimx,modelSettings.dimt,2),1+0j)

    #per-step metropolis accept/reject record, filled by hmcChain
    acceptHistory = np.zeros(metroSteps, dtype=bool)
    tunnelAcceptance = np.zeros(metroSteps, dtype=bool)
        
    for currentStep in tqdm(range(metroSteps), position=tqdmPosition, leave=True):

        gaugeLinks, acceptHistory[currentStep] = hmcStep(modelSettings, gaugeLinks,
                                                          numSubSteps=numSubSteps, rng=rng,cgRtol=cgRtol)

        if(tunneling):
            #if doing tunneling steps, do them here
            gaugeLinks, tunnelAcceptance[currentStep] = tunnelStep(modelSettings, gaugeLinks, rng)

        linkHistory[currentStep] = gaugeLinks
    
    return modelSettings, linkHistory, acceptHistory, tunnelAcceptance