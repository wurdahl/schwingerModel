import numpy as np

def getPlaqAvg(gaugeLinks):
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
    
def plaqStats(burnIn):

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