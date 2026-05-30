import numpy as np
from scipy.special import iv
from scipy.sparse.linalg import cg, bicgstab
from scipy.sparse.linalg import LinearOperator
import scipy.sparse as sparse
from scipy.stats import bootstrap
from scipy.optimize import curve_fit
from tqdm import tqdm


class schwingerModel:

    def __init__(self, dimx = 4, dimt=4, metroSteps = 100, beta = 10, fMass = 1, aSpacing=1,cgRtol = 1e-10, randSeed=0):

        #define gamma matrices
        self.gammax = np.array([[0,1],[1,0]])
        self.gammat = np.array([[0,-1j],[1j,0]])
        
        self.randSeed = randSeed
        self.rng = np.random.default_rng(randSeed)

        self.dimx = dimx
        self.dimt = dimt
        self.metroSteps = metroSteps
        self.beta = beta
        self.fMass = fMass
        self.a = aSpacing
        self.cgRtol = cgRtol
        
        self.gaugeLinks = np.full((dimx,dimt,2),1+0j)

        self.linkHistory = np.zeros((self.metroSteps, self.dimx,self.dimt, 2),dtype="complex128")

        self.storedProps = [None]*self.metroSteps

        #used to store conjugate gradient answers during one trajectory
        self.previous_CG_ans = None

        self.hmcChain()

    def apply_D_vectorized(self, vector=np.full(4*4*2,1+0j), gaugeLinks=np.full((4,4,2),1+0j),dagger=False):
        phi = np.reshape(vector, (self.dimx, self.dimt, 2))
        out = np.zeros_like(phi, dtype=np.complex128)
        
        # 1. Mass term
        out += (self.fMass + 2/self.a) * phi
        
        # 2. Shift fields (periodic boundaries handled automatically by np.roll)
        phi_xp1 = np.roll(phi, shift=-1, axis=0)
        phi_xm1 = np.roll(phi, shift=1, axis=0)
        phi_tp1 = np.roll(phi, shift=-1, axis=1)
        phi_tm1 = np.roll(phi, shift=1, axis=1)
        
        # Shift gauge links for backward directions
        U_x_xm1 = np.roll(gaugeLinks[:, :, 1], shift=1, axis=0)
        U_t_tm1 = np.roll(gaugeLinks[:, :, 0], shift=1, axis=1)
        
        # 3. Time Boundary Conditions (Anti-periodic for fermions)
        bc_fw_t = np.ones((self.dimx, self.dimt, 1))
        bc_fw_t[:, -1, :] = -1.0
        
        bc_bw_t = np.ones((self.dimx, self.dimt, 1))
        bc_bw_t[:, 0, :] = -1.0
        
        # 4. Spinor Matrices
        I = np.eye(2)
        P_plus_x = I + self.gammax
        P_minus_x = I - self.gammax
        P_plus_t = I + self.gammat
        P_minus_t = I - self.gammat
        
        # 5. Kinetic Terms
        # np.einsum('ij,xyj->xyi', Matrix, Field) applies the 2x2 matrix to the spinor at every site x,y
        if(dagger==False):
            # +x direction
            term_xp1 = np.einsum('ij,xyj->xyi', P_minus_x, phi_xp1) * gaugeLinks[:, :, 1, np.newaxis]
            # -x direction
            term_xm1 = np.einsum('ij,xyj->xyi', P_plus_x, phi_xm1) * np.conjugate(U_x_xm1[:, :, np.newaxis])
            # +t direction
            term_tp1 = np.einsum('ij,xyj->xyi', P_minus_t, phi_tp1) * gaugeLinks[:, :, 0, np.newaxis] * bc_fw_t
            # -t direction
            term_tm1 = np.einsum('ij,xyj->xyi', P_plus_t, phi_tm1) * np.conjugate(U_t_tm1[:, :, np.newaxis]) * bc_bw_t
            
        else: #hermitian conjugate
             # +x direction
            term_xp1 = np.einsum('ij,xyj->xyi', P_plus_x, phi_xp1) * gaugeLinks[:, :, 1, np.newaxis]
            # -x direction
            term_xm1 = np.einsum('ij,xyj->xyi', P_minus_x, phi_xm1) * np.conjugate(U_x_xm1[:, :, np.newaxis])
            # +t direction
            term_tp1 = np.einsum('ij,xyj->xyi', P_plus_t, phi_tp1) * gaugeLinks[:, :, 0, np.newaxis] * bc_fw_t
            # -t direction
            term_tm1 = np.einsum('ij,xyj->xyi', P_minus_t, phi_tm1) * np.conjugate(U_t_tm1[:, :, np.newaxis]) * bc_bw_t

        out -= (1/(2*self.a)) * (term_xp1 + term_xm1 + term_tp1 + term_tm1)
        return out.flatten()

    def apply_D_Ddagger(self,vector=np.full(4*4*2,1+0j),gaugeLinks = np.full((4,4,2),1+0j)):
        return self.apply_D_vectorized(self.apply_D_vectorized(vector,gaugeLinks, dagger=True),gaugeLinks)
    

    def pseudoBilinear(self, pseudoField = np.full(4*4*2,1+0j), gaugeLinks = np.full((4,4,2),1+0j)):
        matvec_wrapper = lambda v: self.apply_D_Ddagger(v, gaugeLinks)

        diracOp = LinearOperator((self.dimx*self.dimt*2,self.dimx*self.dimt*2), matvec = matvec_wrapper)

        X, exitcode = bicgstab(diracOp,pseudoField,rtol=self.cgRtol)

        if exitcode != 0:
            raise RuntimeError(f"Conjugate Gradient failed to converge! Exit code: {exitcode}")

        return np.vdot(pseudoField,X).real
    
    def stapleCalc(self,xIndex,tIndex,directionIndex, gaugeLinks):
        x, t, d = xIndex, tIndex, directionIndex
        
        # If link is in the TIME direction (U_t at x,t)
        if d == 0: 
            # Right staple (+x direction loop): U_x(x,t+1) * U_t*(x+1,t) * U_x*(x,t)
            staple_right = (gaugeLinks[x, (t+1)%self.dimt,1] * np.conjugate(gaugeLinks[(x+1)%self.dimx, t,0]) * np.conjugate(gaugeLinks[x, t,1]))
            
            # Left staple (-x direction loop): U_x*(x-1,t+1) * U_t*(x-1,t) * U_x(x-1,t)
            # (Note: Your original code actually had this specific staple mostly right!)
            staple_left = (np.conjugate(gaugeLinks[(x-1)%self.dimx, (t+1)%self.dimt,1]) * np.conjugate(gaugeLinks[(x-1)%self.dimx, t,0]) * gaugeLinks[(x-1)%self.dimx, t,1])
            
            Astaple = staple_right + staple_left

        # If link is in the SPACE direction (U_x at x,t)
        if d == 1: 
            # Top staple (+t direction loop): U_t(x+1,t) * U_x*(x,t+1) * U_t*(x,t)
            staple_top = (gaugeLinks[(x+1)%self.dimx, t,0] * np.conjugate(gaugeLinks[x, (t+1)%self.dimt,1]) * np.conjugate(gaugeLinks[x, t, 0]))
            
            # Bottom staple (-t direction loop): U_t*(x+1,t-1) * U_x*(x,t-1) * U_t(x,t-1)
            staple_bottom = (np.conjugate(gaugeLinks[(x+1)%self.dimx, (t-1)%self.dimt,0]) * np.conjugate(gaugeLinks[x, (t-1)%self.dimt,1]) * gaugeLinks[x, (t-1)%self.dimt,0])
            
            Astaple = staple_top + staple_bottom
        
        return Astaple
    
    def totalAction(self, gaugeLinks):
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
            action = self.beta * np.sum(1.0 - np.real(plaq))
            
            return action

    #calculates the derivative of the action with respect to the generalized coordiantes Q (i.e. the fields)
    #this describes how the conjugate momentum at each link will change during integration
    def hmcForcingFunction(self,gaugeLinks,phis):
        #Force is in the algebra of U(1), so it is always real
        Force = np.zeros((self.dimx,self.dimt, 2))
        #gauge field contribution

        #calculate cg for fermion force calculation
        matvec_wrapper = lambda v: self.apply_D_Ddagger(v, gaugeLinks)

        diracOp = LinearOperator((self.dimx*self.dimt*2,self.dimx*self.dimt*2), matvec = matvec_wrapper)

        #X is (D D^\dagger)\phi
        x0 = self.previous_CG_ans if self.previous_CG_ans is not None else np.zeros_like(phis)
        X, exitcode = cg(diracOp, phis, x0=x0, rtol=self.cgRtol)
        #save X to use on next iteration
        self.previous_CG_ans = X

        if exitcode != 0:
            raise RuntimeError(f"Conjugate Gradient failed to converge! Exit code: {exitcode}")
        
        #Y = D^\dagger X
        Y = self.apply_D_vectorized(X, gaugeLinks,dagger=True)

        Y = np.reshape(Y,(self.dimx,self.dimt,2))
        X = np.reshape(X,(self.dimx,self.dimt,2))

        #identity matrix and constant
        I = np.eye(2, dtype=np.complex128)
        c = 1j / (2 * self.a)

        for x in range(self.dimx):
            for t in range(self.dimt):
                for d in range(2):
                    Astaple = self.stapleCalc(x,t,d,gaugeLinks)
                    Force[x,t,d] = self.beta * np.imag(gaugeLinks[x,t,d]*Astaple)

                #fermion component of the force

                xp1 = (x + 1) % self.dimx
                tp1 = (t + 1) % self.dimt

                #spatial direction
                Z = (np.vdot(X[x,t], -c*(I-self.gammax)*gaugeLinks[x,t,1] @ Y[xp1,t]) 
                                + np.vdot(X[xp1,t], c*(I+self.gammax)*np.conjugate(gaugeLinks[x,t,1]) @ Y[x,t]))
                Force[x,t,1] -=2*Z.real 
                
                #time direction
                Z = (np.vdot(X[x,t], -c*(I-self.gammat)*gaugeLinks[x,t,0] @ Y[x,tp1]) 
                                + np.vdot(X[x,tp1],c*(I+self.gammat)*np.conjugate(gaugeLinks[x,t,0]) @ Y[x,t]))
                
                #enforce antiperiodic boundary condition
                bc_t = -1.0 if t == self.dimt - 1 else 1.0

                Force[x,t,0] -= 2*Z.real * bc_t

        return Force
    
    def hmcForcingFunction_vec(self, gaugeLinks, phis):
        Force = np.zeros((self.dimx, self.dimt, 2))

        # --- CG solve (same as original) ---
        matvec_wrapper = lambda v: self.apply_D_Ddagger(v, gaugeLinks)
        diracOp = LinearOperator((self.dimx*self.dimt*2, self.dimx*self.dimt*2), matvec=matvec_wrapper)
        x0 = self.previous_CG_ans if self.previous_CG_ans is not None else np.zeros_like(phis)
        X, exitcode = cg(diracOp, phis, x0=x0, rtol=self.cgRtol)
        self.previous_CG_ans = X
        if exitcode != 0:
            raise RuntimeError(f"Conjugate Gradient failed to converge! Exit code: {exitcode}")

        Y = self.apply_D_vectorized(X, gaugeLinks, dagger=True)
        Y = np.reshape(Y, (self.dimx, self.dimt, 2))
        X = np.reshape(X, (self.dimx, self.dimt, 2))

        I = np.eye(2, dtype=np.complex128)
        c = 1j / (2 * self.a)

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
        Force[:, :, 0] = self.beta * np.imag(Ut * Astaple_t)

        Ut_tm1     = np.roll(Ut, shift=1,  axis=1)              # Ut[x, t-1]
        Ut_xp1_tm1 = np.roll(Ut_xp1, shift=1, axis=1)          # Ut[x+1, t-1]
        Ux_tm1     = np.roll(Ux, shift=1,  axis=1)              # Ux[x, t-1]

        # Space links: top staple    = Ut[x+1,t]*Ux*[x,t+1]*Ut*[x,t]
        #              bottom staple  = Ut*[x+1,t-1]*Ux*[x,t-1]*Ut[x,t-1]
        Astaple_x = (Ut_xp1 * np.conj(Ux_tp1) * np.conj(Ut)
                     + np.conj(Ut_xp1_tm1) * np.conj(Ux_tm1) * Ut_tm1)
        Force[:, :, 1] = self.beta * np.imag(Ux * Astaple_x)

        # --- Fermion force (vectorized inner products) ---
        P_minus_x = I - self.gammax
        P_plus_x  = I + self.gammax
        P_minus_t = I - self.gammat
        P_plus_t  = I + self.gammat

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
        Pm_t_Y_tp1 = np.einsum('ij,xyj->xyi', P_minus_t, Y_tp1)
        Pp_t_Y     = np.einsum('ij,xyj->xyi', P_plus_t,  Y)
        Z_t = (-c * Ut      * np.einsum('xyi,xyi->xy', np.conj(X),     Pm_t_Y_tp1)
               + c * np.conj(Ut) * np.einsum('xyi,xyi->xy', np.conj(X_tp1), Pp_t_Y))

        # Anti-periodic boundary condition: flip sign at t = dimt-1
        bc_t = np.ones((self.dimx, self.dimt))
        bc_t[:, -1] = -1.0
        Force[:, :, 0] -= 2 * Z_t.real * bc_t

        return Force

    #do one step of an hmc metropolis algorithm
    #returns boolean of success of total step
    #if successful, replaces global value of gaugeLinks
    def hmcStep(self, numSubSteps=100):
        #copy current gauge configuration
        gaugeLinksCopy = np.copy(self.gaugeLinks)

        #used to store conjugate gradient answers during one trajectory
        self.previous_CG_ans = None

        epsilon=1/numSubSteps

        #generate pseduofermions field:
        chi = self.rng.normal(loc=0,scale=1/np.sqrt(2),size=(self.dimx*self.dimt*2))+1j*self.rng.normal(loc=0,scale=1/np.sqrt(2),size=(self.dimx*self.dimt*2))

        phi = self.apply_D_vectorized(chi,self.gaugeLinks)

        #generate initial value for conjugate field
        conjPInitial = self.rng.normal(loc=0,scale=1,size=(self.dimx,self.dimt,2))

        #first momentum half step:
        conjP = conjPInitial - epsilon/2 * self.hmcForcingFunction_vec(gaugeLinksCopy,phi)
        for i in range(numSubSteps-1):
            gaugeLinksCopy *= np.exp((1j)*epsilon *conjP)
            conjP = conjP - epsilon * self.hmcForcingFunction_vec(gaugeLinksCopy,phi)
        #last step
        gaugeLinksCopy *= np.exp((1j)*epsilon *conjP)
        conjP = conjP - epsilon/2 * self.hmcForcingFunction_vec(gaugeLinksCopy,phi)

        metroFactor = np.exp(0.5*np.sum(conjPInitial**2)-0.5*np.sum(conjP**2)
                             +self.totalAction(self.gaugeLinks)-self.totalAction(gaugeLinksCopy)
                             +self.pseudoBilinear(phi,self.gaugeLinks)-self.pseudoBilinear(phi,gaugeLinksCopy))
        r=self.rng.random()
        if(r<metroFactor):
            success=True
            self.gaugeLinks = gaugeLinksCopy
            self.gaugeLinks/= np.abs(self.gaugeLinks)
        else:
            success=False

        return success
    
    def hmcChain(self):
        currentStep = 0
        for currentStep in tqdm(range(self.metroSteps)):
            self.hmcStep()
            self.linkHistory[currentStep] = self.gaugeLinks



    
    
    
    
   
    
    
   
    
    