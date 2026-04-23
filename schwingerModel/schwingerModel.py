import numpy as np
from scipy.special import iv
from scipy.sparse.linalg import cg, bicgstab
from scipy.sparse.linalg import LinearOperator
import scipy.sparse as sparse
from scipy.stats import bootstrap
from scipy.optimize import curve_fit


class schwingerModel:

    def __init__(self, dimx = 4, dimt=4, metroSteps = 100, beta = 10, epsilon=1, fMass = 1, aSpacing=1,cgRtol = 1e-10):

        #define gamma matrices
        self.gammax = np.array([[0,1],[1,0]])
        self.gammat = np.array([[0,-1j],[1j,0]])

        self.rng = np.random.default_rng(0)

        self.dimx = dimx
        self.dimt = dimt
        self.metroSteps = metroSteps
        self.beta = beta
        self.epsilon = epsilon
        self.fMass = fMass
        self.a = aSpacing
        self.cgRtol = cgRtol
        
        self.gaugeLinks = np.full((dimx,dimt,2),1+0j)

        self.accepted = np.zeros(self.metroSteps)
        self.plaqAvgs = np.zeros(self.metroSteps)

        self.linkHistory = np.zeros((self.metroSteps, self.dimx,self.dimt, 2),dtype="complex128")

        self.hmcChain()

    
    def apply_D(self, vector=np.full(4*4*2,1+0j), gaugeLinks=np.full((4,4,2),1+0j)):
        pseudoField = np.reshape(vector, (self.dimx, self.dimt, 2))
        
        outputVector = np.zeros((self.dimx, self.dimt, 2), dtype=np.complex128)
        I = np.diag([1, 1])

        for x in range(self.dimx):
            for t in range(self.dimt):
                xp1 = (x + 1) % self.dimx
                xm1 = (x - 1) % self.dimx
                tp1 = (t + 1) % self.dimt
                tm1 = (t - 1) % self.dimt

                # Anti-periodic boundary conditions for fermions in time
                bc_fw_t = -1.0 if t == self.dimt - 1 else 1.0
                bc_bw_t = -1.0 if t == 0 else 1.0

                # Mass term of Dirac operator
                outputVector[x, t, :] = (self.fMass + 2/self.a) * pseudoField[x, t, :]

                # KINETIC ENERGY TERMS
                
                # +x direction (forward): (I - gammax) * U_x(x,t) * phi(x+1,t)
                # Space links are index 1
                outputVector[x, t, :] -= (1/(2*self.a)) * (I - self.gammax) @ pseudoField[xp1, t, :] * gaugeLinks[x, t, 1]
                
                # -x direction (backward): (I + gammax) * U_x^\dagger(x-1,t) * phi(x-1,t)
                outputVector[x, t, :] -= (1/(2*self.a)) * (I + self.gammax) @ pseudoField[xm1, t, :] * np.conjugate(gaugeLinks[xm1, t, 1])

                # +t direction (forward): (I - gammat) * U_t(x,t) * phi(x,t+1)
                # Time links are index 0
                outputVector[x, t, :] -= bc_fw_t * (1/(2*self.a)) * (I - self.gammat) @ pseudoField[x, tp1, :] * gaugeLinks[x, t, 0]
                
                # -t direction (backward): (I + gammat) * U_t^\dagger(x,t-1) * phi(x,t-1)
                outputVector[x, t, :] -= bc_bw_t * (1/(2*self.a)) * (I + self.gammat) @ pseudoField[x, tm1, :] * np.conjugate(gaugeLinks[x, tm1, 0])

        return outputVector.flatten()

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

    
    def apply_Ddagger(self, vector=np.full(4*4*2,1+0j), gaugeLinks=np.full((4,4,2),1+0j)):
        pseudoField = np.reshape(vector, (self.dimx, self.dimt, 2))
        outputVector = np.zeros((self.dimx, self.dimt, 2), dtype=np.complex128)
        I = np.diag([1, 1])

        for x in range(self.dimx):
            for t in range(self.dimt):
                xp1 = (x + 1) % self.dimx
                xm1 = (x - 1) % self.dimx
                tp1 = (t + 1) % self.dimt
                tm1 = (t - 1) % self.dimt

                bc_fw_t = -1.0 if t == self.dimt - 1 else 1.0
                bc_bw_t = -1.0 if t == 0 else 1.0

                # Mass term remains identical
                outputVector[x, t, :] = (self.fMass + 2/self.a) * pseudoField[x, t, :]

                # EXACT ADJOINT KINETIC TERMS
                # Swaps (I - gamma) with (I + gamma)
                
                # +x direction (forward): Uses (I + gammax)
                outputVector[x, t, :] -= (1/(2*self.a)) * (I + self.gammax) @ pseudoField[xp1, t, :] * gaugeLinks[x, t, 1]
                
                # -x direction (backward): Uses (I - gammax)
                outputVector[x, t, :] -= (1/(2*self.a)) * (I - self.gammax) @ pseudoField[xm1, t, :] * np.conjugate(gaugeLinks[xm1, t, 1])

                # +t direction (forward): Uses (I + gammat)
                outputVector[x, t, :] -= bc_fw_t * (1/(2*self.a)) * (I + self.gammat) @ pseudoField[x, tp1, :] * gaugeLinks[x, t, 0]
                
                # -t direction (backward): Uses (I - gammat)
                outputVector[x, t, :] -= bc_bw_t * (1/(2*self.a)) * (I - self.gammat) @ pseudoField[x, tm1, :] * np.conjugate(gaugeLinks[x, tm1, 0])

        return outputVector.flatten()
    
    
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
        X, exitcode = cg(diracOp,phis,rtol=self.cgRtol)

        if exitcode != 0:
            raise RuntimeError(f"Conjugate Gradient failed to converge! Exit code: {exitcode}")
        
        #Y = D^\dagger X
        Y = self.apply_Ddagger(X, gaugeLinks)

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
    
    #do one step of an hmc metropolis algorithm
    #returns boolean of success of total step
    #if successful, replaces global value of gaugeLinks
    def hmcStep(self, numSubSteps=100):
        #copy current gauge configuration
        gaugeLinksCopy = np.copy(self.gaugeLinks)

        epsilon=1/numSubSteps

        #generate pseduofermions field:
        chi = self.rng.normal(loc=0,scale=1/np.sqrt(2),size=(self.dimx*self.dimt*2))+1j*self.rng.normal(loc=0,scale=1/np.sqrt(2),size=(self.dimx*self.dimt*2))

        phi = self.apply_D(chi,self.gaugeLinks)

        #generate initial value for conjugate field
        conjPInitial = self.rng.normal(loc=0,scale=1,size=(self.dimx,self.dimt,2))

        #first momentum half step:
        conjP = conjPInitial - epsilon/2 * self.hmcForcingFunction(gaugeLinksCopy,phi)
        for i in range(numSubSteps-1):
            gaugeLinksCopy *= np.exp((1j)*epsilon *conjP)
            conjP = conjP - epsilon * self.hmcForcingFunction(gaugeLinksCopy,phi)
        #last step
        gaugeLinksCopy *= np.exp((1j)*epsilon *conjP)
        conjP = conjP - epsilon/2 * self.hmcForcingFunction(gaugeLinksCopy,phi)

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
        for currentStep in range(self.metroSteps):
            #if the hmc step is successful, move forward

            self.accepted[currentStep] = self.hmcStep()
            self.plaqAvgs[currentStep] = self.getPlaqAvg(self.gaugeLinks)
            self.linkHistory[currentStep] = self.gaugeLinks

            if(currentStep%50==0):
                print("Current step",currentStep)


    def getPlaqAvg(self, gaugeLinks):
        Ut = gaugeLinks[:,:,0] # Time links (shape: dimx, dimt)
        Ux = gaugeLinks[:,:,1] # Space links (shape: dimx, dimt)
        
        # Shift arrays to get U_t(x+1, t) and U_x(x, t+1)
        # axis=0 corresponds to the space dimension (dimx)
        # axis=1 corresponds to the time dimension (dimt)
        Ut_shifted_x = np.roll(Ut, shift=-1, axis=0) 
        Ux_shifted_t = np.roll(Ux, shift=-1, axis=1) 
        
        # Multiply the four sides of the plaquette
        plaq = Ux * Ut_shifted_x * np.conjugate(Ux_shifted_t) * np.conjugate(Ut)
        
        return np.mean(np.real(plaq))
    
    def plaqStats(self,burnIn):

        burnedInAvgs = self.plaqAvgs[burnIn:]
    
        return np.array([np.mean(burnedInAvgs),np.std(burnedInAvgs)/np.sqrt(len(burnedInAvgs))])
    
    
    def correlStats(self,burnIn,autocorrSkip=1, Gamma=np.array([[1j,0],[0,-1j]]), includeDisc = True):

        acceptedCorrel_conn = []
        acceptedCorrel_disc = []
        source_trace = []

        for i in range(burnIn,self.metroSteps,autocorrSkip):
            Cconn, Cdisc, sTrace = self.getCorrelation(self.linkHistory[i],Gamma)
            
            acceptedCorrel_conn.append(Cconn)
            acceptedCorrel_disc.append(Cdisc)
            source_trace.append(sTrace)

        acceptedCorrel_conn = np.array(acceptedCorrel_conn)
        acceptedCorrel_disc = np.array(acceptedCorrel_disc)
        source_trace = np.array(source_trace)

        mean_vev = np.mean(source_trace)

        if(includeDisc):
            totalCorrels = acceptedCorrel_conn + acceptedCorrel_disc - self.dimx *(mean_vev**2)
        else:
            totalCorrels = acceptedCorrel_conn

        totalCorrelMean = np.mean(totalCorrels,axis=0)

        totalCorrelErr = np.zeros((self.dimt,2))

        for i in range(self.dimt):
            bootstrapRes = bootstrap((totalCorrels[:,i],), np.mean, confidence_level=.95, rng=self.rng)
            totalCorrelErr[i] = np.abs(bootstrapRes.confidence_interval-totalCorrelMean[i])
        
        return [totalCorrelMean,totalCorrelErr]
    
    def effectiveMassStats(self,burnIn,autocorrSkip=1, Gamma=np.array([[1j,0],[0,-1j]]), includeDisc = True):
        acceptedCorrel_conn = []
        acceptedCorrel_disc = []
        source_trace = []

        for i in range(burnIn,self.metroSteps,autocorrSkip):
            Cconn, Cdisc, sTrace = self.getCorrelation(self.linkHistory[i],Gamma)
            
            acceptedCorrel_conn.append(Cconn)
            acceptedCorrel_disc.append(Cdisc)
            source_trace.append(sTrace)

        acceptedCorrel_conn = np.array(acceptedCorrel_conn)
        acceptedCorrel_disc = np.array(acceptedCorrel_disc)
        source_trace = np.array(source_trace)

        mean_vev = np.mean(source_trace)

        if(includeDisc):
            totalCorrels = acceptedCorrel_conn + acceptedCorrel_disc - self.dimx *(mean_vev**2)
        else:
            totalCorrels = acceptedCorrel_conn

        #find effective mass curves for each of these correlations

        numerators = totalCorrels[:,2:] + totalCorrels[:,:-2]
        denominators = 2 * totalCorrels[:,1:-1]

        # 3. Calculate the cosh-based effective mass
        effectiveMass = np.arccosh(numerators / denominators)

        effectiveMassMean = np.mean(effectiveMass,axis=0)

        effectiveMassErr = np.zeros((self.dimt-2,2))

        for i in range(self.dimt-2):
            bootstrapRes = bootstrap((effectiveMass[:,i],), np.mean, confidence_level=.95, rng=self.rng)
            effectiveMassErr[i] = np.abs(bootstrapRes.confidence_interval - effectiveMassMean[i])
        
        return [effectiveMassMean,effectiveMassErr]
    
    def coshMassProfile(self, nt, E_0, A_0):
        return A_0* np.cosh((nt-self.dimt)/2 *E_0)
    
    def estMass(self,burnIn,autocorrSkip=1, Gamma=np.array([[1j,0],[0,-1j]]), includeDisc = True):

        acceptedCorrel_conn = []
        acceptedCorrel_disc = []
        source_trace = []

        for i in range(burnIn,self.metroSteps,autocorrSkip):
            Cconn, Cdisc, sTrace = self.getCorrelation(self.linkHistory[i],Gamma)
            
            acceptedCorrel_conn.append(Cconn)
            acceptedCorrel_disc.append(Cdisc)
            source_trace.append(sTrace)

        acceptedCorrel_conn = np.array(acceptedCorrel_conn)
        acceptedCorrel_disc = np.array(acceptedCorrel_disc)
        source_trace = np.array(source_trace)

        mean_vev = np.mean(source_trace)

        if(includeDisc):
            totalCorrels = acceptedCorrel_conn + acceptedCorrel_disc - self.dimx *(mean_vev**2)
        else:
            totalCorrels = acceptedCorrel_conn


        covMat = np.cov(totalCorrels,rowvar=False)

        fit = curve_fit(self.coshMassProfile, xdata=np.arange(self.dimt), 
                        ydata = np.mean(totalCorrels,axis=0),
                        sigma=covMat,absolute_sigma=True,
                        bounds = ([.1,0],[3,1E-2]))
        
        return fit
        
    
    #builds the dirac operator using the global gaugeLinks configuration
    # matrix is a square matrix with dimensional ordering (space, time, spin) 
    def buildDiracOp(self, gaugeLinks):
        #dirac dimensions
        dimD = 2
        eyeD = np.eye(dimD)

        shift_x_1Dpos = np.roll(np.eye(self.dimx), -1, axis=0) # This is \delta_{x_n+1, x_m}
        shift_t_1Dpos = np.roll(np.eye(self.dimt), -1, axis=0)
        shift_x_1Dneg = np.roll(np.eye(self.dimx), +1, axis=0) # This is \delta_{x_n+1, x_m}
        shift_t_1Dneg = np.roll(np.eye(self.dimt), +1, axis=0)
        time_identity = np.eye(self.dimt)                      # This is \delta_{t_n, t_m}
        space_identity = np.eye(self.dimx)

        #anti-periodic boundary conditions for fermions in time
        shift_t_1Dpos[self.dimt - 1, 0] = -1.0
        shift_t_1Dneg[0, self.dimt - 1] = -1.0

        #space-time shift operators
        T_x_pos = sparse.kron(shift_x_1Dpos, time_identity)
        T_x_neg = sparse.kron(shift_x_1Dneg, time_identity)
        T_t_pos = sparse.kron(space_identity, shift_t_1Dpos)
        T_t_neg = sparse.kron(space_identity, shift_t_1Dneg)

        #flattened gaugelinks
        spaceLinks = np.diag(gaugeLinks[:,:,1].flatten())
        timeLinks = np.diag(gaugeLinks[:,:,0].flatten())

        #start building dirac operator matrix
        Dee = (self.fMass+2/self.a)*sparse.kron(np.eye(self.dimx), sparse.kron(np.eye(self.dimt),eyeD))
        #positive shifts
        Dee-=1/(2*self.a) * sparse.kron(spaceLinks@T_x_pos, eyeD-self.gammax)
        Dee-=1/(2*self.a) * sparse.kron(timeLinks@T_t_pos, eyeD-self.gammat)
        #negative shifts
        Dee-=1/(2*self.a) * sparse.kron(T_x_neg@np.conj(spaceLinks),eyeD+self.gammax)
        Dee-=1/(2*self.a) * sparse.kron(T_t_neg@np.conj(timeLinks),eyeD+self.gammat)

        return Dee
    
    def getCorrelation(self,gaugeLinks,Gamma=np.array([[1j,0],[0,-1j]])):

        dOp = self.buildDiracOp(gaugeLinks)
        prop = np.linalg.inv(dOp.toarray())

        stridex = self.dimt*2
        stridet = 2

        correl_conn = np.zeros(self.dimt)
        correl_disc = np.zeros(self.dimt)


        #t_n/x_n will always be set at zero for now

        t_n=0
        x_n=0

        # Store the local trace Tr[Gamma S(x,t; x,t)] for every point
        local_traces = np.zeros((self.dimt, self.dimx), dtype=complex)

        for t in range(self.dimt):           
            for x in range(self.dimx):
                idx_start = x * stridex + t * stridet
                idx_end = idx_start + 2
                propxx = prop[idx_start:idx_end, idx_start:idx_end]
                local_traces[t, x] = np.trace(Gamma @ propxx)

        trace_source_avg = np.mean(local_traces).real

        # Pre-sum the sink traces over spatial sites for the disconnected part
        # This gives an array of length dimt
        spatial_summed_traces = np.sum(local_traces, axis=1)

        #loop over changes in time
        for delta_t in range(self.dimt):
           
            #Diconnected part
            disc_sum = 0
            #shift to all times to collect all possible products for disconnected part
            for t_src in range(self.dimt):
                t_sink = (t_src+delta_t)%self.dimt
                disc_sum+= (spatial_summed_traces[t_sink]*spatial_summed_traces[t_src]).real
            correl_disc[delta_t] = disc_sum / (self.dimt*self.dimx)

            #connected part
            t_n = 0
            x_n = 0
            idx_n_start = x_n * stridex + t_n * stridet
            idx_n_end = idx_n_start + 2

            for x_m in range(self.dimx):

                idx_m_start = x_m * stridex + delta_t * stridet
                idx_m_end = idx_m_start + 2

                #connected part of the correlation
                propnm = prop[idx_n_start:idx_n_end, idx_m_start:idx_m_end]
                propmn = prop[idx_m_start:idx_m_end, idx_n_start:idx_n_end]

                correl_conn[delta_t]+=-np.trace(Gamma@propnm@Gamma@propmn).real
        
        #return both connected and disconnected part
        return correl_conn, correl_disc, trace_source_avg