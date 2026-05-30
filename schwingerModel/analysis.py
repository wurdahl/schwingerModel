import numpy as np
import scipy.sparse as sparse
from scipy.stats import bootstrap
from scipy.optimize import curve_fit
from tqdm import tqdm

from .schwingerModel import schwingerModel
from . import buildOps as ops
from . import correlation as corr

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


def correlStats(modelObj: schwingerModel, burnIn,autocorrSkip=1, Gamma=np.array([[1j,0],[0,-1j]]),
                 includeDisc = False, chemicalPot = 0, k=0,
                   smearing=False, kappa=.1, smearingSteps=1):
    """Compute the ensemble-averaged meson correlator with bootstrap errors.

    Loops over gauge configurations (skipping burn-in and thinning by
    autocorrSkip), evaluates the connected (and optionally disconnected)
    correlator on each, then returns the weighted mean and 95% bootstrap
    confidence interval.

    Parameters
    ----------
    modelObj : schwingerModel
        Lattice model containing gauge link history and parameters.
    burnIn : int
        Number of initial configurations to discard as thermalization.
    autocorrSkip : int
        Stride between configurations to reduce autocorrelations.
    Gamma : (2,2) complex array
        Dirac structure at source and sink.
    includeDisc : bool
        If True, add the disconnected diagram and subtract the VEV squared.
    chemicalPot : float
        Chemical potential for phase-reweighting. Zero disables reweighting.
    k : int
        Spatial momentum mode for projection. If nonzero, each configuration
        is evaluated at both +k and -k and the weights are doubled accordingly.
    smearing : bool
        If True, apply Jacobi smearing to source and sink via smearedPropagator.
    kappa : float
        Jacobi smearing weight parameter.
    smearingSteps : int
        Number of Jacobi smearing steps (power-series order).

    Returns
    -------
    [mean, errors] where
        mean   : (dimt,) array, real part of the weighted-average correlator.
        errors : (2, dimt) array, [upper, lower] 95% bootstrap CI half-widths.
    """
    acceptedCorrel_conn = []
    acceptedCorrel_disc = []
    source_trace = []

    #weights for chemicalPot
    weightsMu = getWeightingFactors(modelObj, chemicalPot, burnIn,  autocorrSkip)

    #if k!=0, then each config will show up twice, so we need to repeat the weights
    if(k!=0):
        weightsMu = np.repeat(weightsMu,2)

    for i in tqdm(range(burnIn,modelObj.metroSteps,autocorrSkip)):
        Cconn, Cdisc, sTrace = getCorrelation(modelObj, modelObj.linkHistory[i],Gamma, k=k,
                                              smearing=smearing, kappa=kappa, smearingSteps=smearingSteps)
        
        acceptedCorrel_conn.append(Cconn)
        acceptedCorrel_disc.append(Cdisc)
        source_trace.append(sTrace)

        if(k!=0):
            Cconn, Cdisc, sTrace = getCorrelation(modelObj, modelObj.linkHistory[i],Gamma, k=-k,
                                              smearing=smearing, kappa=kappa, smearingSteps=smearingSteps)
        
            acceptedCorrel_conn.append(Cconn)
            acceptedCorrel_disc.append(Cdisc)
            source_trace.append(sTrace)


    acceptedCorrel_conn = np.array(acceptedCorrel_conn)
    acceptedCorrel_disc = np.array(acceptedCorrel_disc)
    source_trace = np.array(source_trace)

    mean_vev = np.average(source_trace, axis=0, weights=weightsMu)

    if(includeDisc):
        totalCorrels = acceptedCorrel_conn + acceptedCorrel_disc - modelObj.dimx *(mean_vev*np.conjugate(mean_vev))
    else:
        totalCorrels = acceptedCorrel_conn

    totalCorrelMean = np.real(np.average(totalCorrels,axis=0,weights=weightsMu))

    #bootstrapping
    numResamples = 10000
    rng = np.random.default_rng()

    resamples = rng.choice(len(totalCorrels), size=(numResamples, len(totalCorrels)))

    # (numResamples, n_configs, dimt) and (numResamples, n_configs)
    correl_boot = totalCorrels[resamples]
    w_boot = weightsMu[resamples]

    # weighted mean for each bootstrap sample -> (numResamples, dimt)
    bootstrap_means = np.real(
        np.sum(correl_boot * w_boot[:, :, np.newaxis], axis=1) /
        np.sum(w_boot, axis=1, keepdims=True)
    )

    low  = np.percentile(bootstrap_means, 2.5,  axis=0)
    high = np.percentile(bootstrap_means, 97.5, axis=0)

    return [totalCorrelMean, np.array([high-totalCorrelMean, totalCorrelMean-low])]

def effectiveMassStats(modelObj,burnIn,autocorrSkip=1, Gamma=np.array([[1j,0],[0,-1j]]), includeDisc = True, coshExpr = True, cleanNans = False, chemicalPot = 0):
    acceptedCorrel_conn = []
    acceptedCorrel_disc = []
    source_trace = []

    #weights for chemicalPot

    weightsMu = getWeightingFactors(modelObj,chemicalPot, burnIn,  autocorrSkip)

    for i in tqdm(range(burnIn,modelObj.metroSteps,autocorrSkip)):
        Cconn, Cdisc, sTrace = getCorrelation(modelObj,modelObj.linkHistory[i],Gamma)
        
        acceptedCorrel_conn.append(Cconn)
        acceptedCorrel_disc.append(Cdisc)
        source_trace.append(sTrace)

    acceptedCorrel_conn = np.array(acceptedCorrel_conn)
    acceptedCorrel_disc = np.array(acceptedCorrel_disc)
    source_trace = np.array(source_trace)

    mean_vev = np.average(source_trace, axis=0, weights=weightsMu)

    if(includeDisc):
        totalCorrels = acceptedCorrel_conn + acceptedCorrel_disc - modelObj.dimx *(mean_vev*np.conjugate(mean_vev))
    else:
        totalCorrels = acceptedCorrel_conn

    #find effective mass curves for each of these correlations

    if(coshExpr):
        numerators = totalCorrels[:,2:] + totalCorrels[:,:-2]
        denominators = 2 * totalCorrels[:,1:-1]

        effectiveMass = np.arccosh(numerators / denominators)
    else:
        effectiveMass = np.log(totalCorrels[:,:-1]/totalCorrels[:,1:])

    effectiveMassMean = np.real(np.average(effectiveMass,axis=0,weights=weightsMu))

    #bootstrapping
    numResamples = 10000
    rng = np.random.default_rng()

    resamples = rng.choice(len(effectiveMass), size=(numResamples, len(effectiveMass)))

    # (numResamples, n_configs, dimt) and (numResamples, n_configs)
    effectiveMass = effectiveMass[resamples]
    w_boot = weightsMu[resamples]

    # weighted mean for each bootstrap sample -> (numResamples, dimt)
    bootstrap_means = np.real(
        np.sum(effectiveMass * w_boot[:, :, np.newaxis], axis=1) /
        np.sum(w_boot, axis=1, keepdims=True)
    )

    low  = np.percentile(bootstrap_means, 2.5,  axis=0)
    high = np.percentile(bootstrap_means, 97.5, axis=0)

    return [effectiveMassMean, np.array([high-effectiveMassMean, effectiveMassMean-low])]

def effectiveMassProp(correlStats, coshExpr=False):    
    #extract Correlation function
    cFunc = correlStats[0]
    cFuncErr = np.mean(correlStats[1],axis=0)

    if(not coshExpr):
        effectiveMass = np.log(np.array(cFunc[:-1])/np.array(cFunc[1:]))

        cFuncFracError = cFuncErr/cFunc

        effectiveMassErr = np.sqrt(cFuncFracError[:-1]**2+cFuncFracError[1:]**2)
    else:
        x1=cFunc[2:]
        x2=cFunc[1:-1]
        x3=cFunc[:-2]

        numerators = x1 + x3
        denominators = 2 * x2

        effectiveMass = np.arccosh(numerators / denominators)

        dx1Sq=cFuncErr[2:]**2
        dx2Sq=cFuncErr[1:-1]**2
        dx3Sq=cFuncErr[:-2]**2
        
        numerators = dx1Sq+(dx3Sq* x2**2+ dx2Sq*(x1+x3)**2)/x2**2
        denominators = (x1-2*x2+x3)*(x1+2*x2+x3)

        effectiveMassErr = np.sqrt(numerators/denominators)

    return [effectiveMass,effectiveMassErr]


def numDensityStats(modelObj, burnIn, autocorrSkip=1, chemicalPot=0.0):
    V = modelObj.a**2*modelObj.dimx * modelObj.dimt

    # reweighting factors for finite mu
    #weights for chemicalPot

    weights = getWeightingFactors(modelObj,chemicalPot, burnIn,  autocorrSkip)

    #get a sense of where reweighting is valid
    validity = (np.abs(np.average(weights))/np.average(np.abs(weights)))

    n_mu_samples = []
    n_0_samples  = []   # vacuum subtraction

    for i in tqdm(range(burnIn, modelObj.metroSteps, autocorrSkip)):
        links = modelObj.linkHistory[i]

        #build dirac props
        S_mu = np.linalg.inv(ops.buildDiracOp(modelObj, links, chemicalPot).toarray())
        S_0  = np.linalg.inv(ops.buildDiracOp(modelObj, links, 0.0).toarray())

        #build number density operators
        n_mu = ops.buildNumberDensOp(modelObj, links, chemicalPot).toarray()
        n_0 = ops.buildNumberDensOp(modelObj, links, 0.0).toarray()

        n_mu_samples.append(np.trace(S_mu@n_mu)/V)
        n_0_samples.append(np.trace(S_0@n_0)/V)

    n_mu_samples = np.array(n_mu_samples)
    n_0_samples  = np.array(n_0_samples)

    # correlated vacuum subtraction (same gauge config) reduces variance

    mean = np.average(n_mu_samples, weights=weights) - np.mean(n_0_samples)

    # bootstrap error
    rng = np.random.default_rng()
    N = len(n_mu_samples)
    numResamples=10000
    resamples = rng.choice(N, size=(numResamples, N))

    n_mu_boot = n_mu_samples[resamples]
    n_0_boot = n_0_samples[resamples]
    w_boot = weights[resamples]

    bootstrap_means = np.real(
        np.sum(n_mu_boot * w_boot, axis=1) /
        np.sum(w_boot, axis=1) - np.mean(n_0_boot,axis=1)
    )

    low  = np.percentile(bootstrap_means, 2.5,  axis=0)
    high = np.percentile(bootstrap_means, 97.5, axis=0)

    return [np.real(mean), np.real(np.array([high-mean, mean-low])), validity]


def getCorrelation(modelObj,gaugeLinks,Gamma=np.array([[1j,0],[0,-1j]]), k=0, smearing=False, kappa= .1, smearingSteps=1):

    if(smearing):
        prop = ops.smearedPropagator(modelObj, gaugeLinks, kappa, smearingSteps)
    else:
        dOp = ops.buildDiracOp(modelObj, gaugeLinks)

        prop = np.linalg.inv(dOp.toarray())

    stridex = modelObj.dimt*2
    stridet = 2

    correl_conn = np.zeros(modelObj.dimt,dtype=np.complex128)
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

            #include momentum projection
            correl_conn[delta_t]+=-np.trace(Gamma@propnm@Gamma@propmn)*np.exp(-1j*2*np.pi*k*x_m/modelObj.dimx)
    
    #return both connected and disconnected part
    return np.real(correl_conn), correl_disc, trace_source_avg

def getWeightingFactors(modelObj: schwingerModel, chemicalPot= 1,burnIn=1, autocorrSkip=10):
    #loop through gaugeLinks of the modelObj and get the weightings:

    if(chemicalPot==0):
        return np.ones(len(np.arange(burnIn,modelObj.metroSteps,autocorrSkip)))

    weights = []

    for i in range(burnIn,modelObj.metroSteps,autocorrSkip):
        currLinks = modelObj.linkHistory[i]
        dOp = ops.buildDiracOp(modelObj, currLinks).toarray()
        dOpmu = ops.buildDiracOp(modelObj, currLinks, chemicalPot).toarray()

        sign_0, logdet_0 = np.linalg.slogdet(dOp)
        sign_mu, logdet_mu = np.linalg.slogdet(dOpmu)
        weights.append((sign_mu / sign_0) * np.exp(logdet_mu - logdet_0))

    #need to square the final weights because there are two degenerate fermions in the problem.
    return np.array(weights)**2


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