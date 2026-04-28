import numpy as np
import scipy.sparse as sparse
from scipy.stats import bootstrap
from scipy.optimize import curve_fit
from tqdm import tqdm

from .schwingerModel import schwingerModel

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
    
def plaqStats(modelObj: schwingerModel, burnIn=1):

    #loop over all configs in modelObj to get plaq averages
    plaqAvgs = np.zeros(modelObj.metroSteps)

    for i in range(modelObj.metroSteps):
        plaqAvgs[i] = getPlaqAvg(modelObj.linkHistory[i])
    
    burnedInAvgs = plaqAvgs[burnIn:]

    return np.array([np.mean(burnedInAvgs),np.std(burnedInAvgs)/np.sqrt(len(burnedInAvgs))])


def correlStats(modelObj,burnIn,autocorrSkip=1, Gamma=np.array([[1j,0],[0,-1j]]), includeDisc = True):

    acceptedCorrel_conn = []
    acceptedCorrel_disc = []
    source_trace = []

    for i in tqdm(range(burnIn,modelObj.metroSteps,autocorrSkip)):
        Cconn, Cdisc, sTrace = getCorrelation(modelObj, modelObj.linkHistory[i],Gamma)
        
        acceptedCorrel_conn.append(Cconn)
        acceptedCorrel_disc.append(Cdisc)
        source_trace.append(sTrace)

    acceptedCorrel_conn = np.array(acceptedCorrel_conn)
    acceptedCorrel_disc = np.array(acceptedCorrel_disc)
    source_trace = np.array(source_trace)

    mean_vev = np.mean(source_trace)

    if(includeDisc):
        totalCorrels = acceptedCorrel_conn + acceptedCorrel_disc - modelObj.dimx *(mean_vev**2)
    else:
        totalCorrels = acceptedCorrel_conn

    totalCorrelMean = np.mean(totalCorrels,axis=0)

    totalCorrelErr = np.zeros((modelObj.dimt,2))

    for i in range(modelObj.dimt):
        bootstrapRes = bootstrap((totalCorrels[:,i],), np.mean, confidence_level=.95, rng=modelObj.rng)
        totalCorrelErr[i] = np.abs(bootstrapRes.confidence_interval-totalCorrelMean[i])
    
    return [totalCorrelMean,totalCorrelErr]

def effectiveMassStats(modelObj,burnIn,autocorrSkip=1, Gamma=np.array([[1j,0],[0,-1j]]), includeDisc = True, coshExpr = True):
    acceptedCorrel_conn = []
    acceptedCorrel_disc = []
    source_trace = []

    for i in tqdm(range(burnIn,modelObj.metroSteps,autocorrSkip)):
        Cconn, Cdisc, sTrace = getCorrelation(modelObj,modelObj.linkHistory[i],Gamma)
        
        acceptedCorrel_conn.append(Cconn)
        acceptedCorrel_disc.append(Cdisc)
        source_trace.append(sTrace)

    acceptedCorrel_conn = np.array(acceptedCorrel_conn)
    acceptedCorrel_disc = np.array(acceptedCorrel_disc)
    source_trace = np.array(source_trace)

    mean_vev = np.mean(source_trace)

    if(includeDisc):
        totalCorrels = acceptedCorrel_conn + acceptedCorrel_disc - modelObj.dimx *(mean_vev**2)
    else:
        totalCorrels = acceptedCorrel_conn

    #find effective mass curves for each of these correlations

    if(coshExpr):
        numerators = totalCorrels[:,2:] + totalCorrels[:,:-2]
        denominators = 2 * totalCorrels[:,1:-1]

        effectiveMass = np.arccosh(numerators / denominators)
    else:
        effectiveMass = np.log(totalCorrels[:,:-1]/totalCorrels[:,1:])

    effectiveMassMean = np.mean(effectiveMass,axis=0)

    effectiveMassErr = np.zeros((len(effectiveMassMean),2))

    for i in range(len(effectiveMassMean)):
        bootstrapRes = bootstrap((effectiveMass[:,i],), np.mean, confidence_level=.95, rng=modelObj.rng)
        effectiveMassErr[i] = np.abs(bootstrapRes.confidence_interval - effectiveMassMean[i])
    
    return [effectiveMassMean,effectiveMassErr]


    
 #builds the dirac operator using the global gaugeLinks configuration
# matrix is a square matrix with dimensional ordering (space, time, spin) 
def buildDiracOp(modelObj, gaugeLinks):
    #dirac dimensions
    dimD = 2
    eyeD = np.eye(dimD)

    shift_x_1Dpos = np.roll(np.eye(modelObj.dimx), -1, axis=0) # This is \delta_{x_n+1, x_m}
    shift_t_1Dpos = np.roll(np.eye(modelObj.dimt), -1, axis=0)
    shift_x_1Dneg = np.roll(np.eye(modelObj.dimx), +1, axis=0) # This is \delta_{x_n+1, x_m}
    shift_t_1Dneg = np.roll(np.eye(modelObj.dimt), +1, axis=0)
    time_identity = np.eye(modelObj.dimt)                      # This is \delta_{t_n, t_m}
    space_identity = np.eye(modelObj.dimx)

    #anti-periodic boundary conditions for fermions in time
    shift_t_1Dpos[modelObj.dimt - 1, 0] = -1.0
    shift_t_1Dneg[0, modelObj.dimt - 1] = -1.0

    #space-time shift operators
    T_x_pos = sparse.kron(shift_x_1Dpos, time_identity)
    T_x_neg = sparse.kron(shift_x_1Dneg, time_identity)
    T_t_pos = sparse.kron(space_identity, shift_t_1Dpos)
    T_t_neg = sparse.kron(space_identity, shift_t_1Dneg)

    #flattened gaugelinks
    spaceLinks = np.diag(gaugeLinks[:,:,1].flatten())
    timeLinks = np.diag(gaugeLinks[:,:,0].flatten())

    #start building dirac operator matrix
    Dee = (modelObj.fMass+2/modelObj.a)*sparse.kron(np.eye(modelObj.dimx), sparse.kron(np.eye(modelObj.dimt),eyeD))
    #positive shifts
    Dee-=1/(2*modelObj.a) * sparse.kron(spaceLinks@T_x_pos, eyeD-modelObj.gammax)
    Dee-=1/(2*modelObj.a) * sparse.kron(timeLinks@T_t_pos, eyeD-modelObj.gammat)
    #negative shifts
    Dee-=1/(2*modelObj.a) * sparse.kron(T_x_neg@np.conj(spaceLinks),eyeD+modelObj.gammax)
    Dee-=1/(2*modelObj.a) * sparse.kron(T_t_neg@np.conj(timeLinks),eyeD+modelObj.gammat)

    return Dee

def getCorrelation(modelObj,gaugeLinks,Gamma=np.array([[1j,0],[0,-1j]])):

    dOp = buildDiracOp(modelObj, gaugeLinks)
    prop = np.linalg.inv(dOp.toarray())

    stridex = modelObj.dimt*2
    stridet = 2

    correl_conn = np.zeros(modelObj.dimt)
    correl_disc = np.zeros(modelObj.dimt)

    #t_n/x_n will always be set at zero for now

    t_n=0
    x_n=0

    # Store the local trace Tr[Gamma S(x,t; x,t)] for every point
    local_traces = np.zeros((modelObj.dimt, modelObj.dimx), dtype=complex)

    for t in range(modelObj.dimt):           
        for x in range(modelObj.dimx):
            idx_start = x * stridex + t * stridet
            idx_end = idx_start + 2
            propxx = prop[idx_start:idx_end, idx_start:idx_end]
            local_traces[t, x] = np.trace(Gamma @ propxx)

    trace_source_avg = np.mean(local_traces).real

    # Pre-sum the sink traces over spatial sites for the disconnected part
    # This gives an array of length dimt
    spatial_summed_traces = np.sum(local_traces, axis=1)

    #loop over changes in time
    for delta_t in range(modelObj.dimt):
        
        #Diconnected part
        disc_sum = 0
        #shift to all times to collect all possible products for disconnected part
        for t_src in range(modelObj.dimt):
            t_sink = (t_src+delta_t)%modelObj.dimt
            disc_sum+= (spatial_summed_traces[t_sink]*spatial_summed_traces[t_src]).real
        correl_disc[delta_t] = disc_sum / (modelObj.dimt*modelObj.dimx)

        #connected part
        t_n = 0
        x_n = 0
        idx_n_start = x_n * stridex + t_n * stridet
        idx_n_end = idx_n_start + 2

        for x_m in range(modelObj.dimx):

            idx_m_start = x_m * stridex + delta_t * stridet
            idx_m_end = idx_m_start + 2

            #connected part of the correlation
            propnm = prop[idx_n_start:idx_n_end, idx_m_start:idx_m_end]
            propmn = prop[idx_m_start:idx_m_end, idx_n_start:idx_n_end]

            correl_conn[delta_t]+=-np.trace(Gamma@propnm@Gamma@propmn).real
    
    #return both connected and disconnected part
    return correl_conn, correl_disc, trace_source_avg


def getEffMassRhoBar(modelObj: schwingerModel):
    allCorrs = []
    nSamp=modelObj.metroSteps
    for i in range(nSamp):
        allCorrs.append(getCorrelation(modelObj,modelObj.linkHistory[i])[0])

    allCorrs = np.array(allCorrs)
    effectiveMassExample = np.log(allCorrs[:,4]/allCorrs[:,5])

    np.arange(nSamp)

    GammaBar = 1/(nSamp - np.arange(nSamp))*(np.correlate(effectiveMassExample-np.mean(effectiveMassExample),effectiveMassExample-np.mean(effectiveMassExample),mode='full')[nSamp-1:])
    rhoBar = GammaBar/GammaBar[0]

    return rhoBar

def get_integrated_autocorr_time_statistical(rho_bar, N_conf):
    """
    Extracts the integrated autocorrelation time (tau_int) and its error 
    using the statistical noise threshold method (Equations 26, 27, and 28).
    
    Parameters:
    -----------
    rho_bar : numpy array or list
        The normalized autocorrelation function rho(tau) starting at lag 0.
    N_conf : int
        The total number of configurations (HMC trajectories) used to calculate rho_bar.
        
    Returns:
    --------
    tau_int : float
        The integrated autocorrelation time.
    error_tau_int : float
        The statistical error on tau_int.
    W : int
        The truncation window where the sum was cut off.
    """
    tau_int = 0.5
    W = 0
    
    # Loop through fictitious time tau, starting at lag 1
    for tau in range(1, len(rho_bar)):
        tau_int += rho_bar[tau]
        W = tau
        
        # Calculate the statistical noise of rho_bar(tau)
        # Using the standard approximation: variance ~ 2 * tau_int / N_conf
        noise_threshold = np.sqrt(2 * tau_int / N_conf)
        
        # Equation 27: Stop if the signal rho_bar(tau) falls below the noise
        # (Note: If rho_bar goes negative, it will also correctly trigger this break)
        if rho_bar[tau] <= noise_threshold:
            break
            
    else:
        # This executes if the loop finishes without breaking
        print("Warning: Windowing condition not met. N_conf may be too small.")

    # Equation 28: Calculate the error of integrated autocorrelation time
    variance_tau_int = ((4 * W + 2) / N_conf) * (tau_int**2)
    error_tau_int = np.sqrt(variance_tau_int)
    
    return tau_int, error_tau_int, W