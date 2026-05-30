import numpy as np
import scipy.sparse as sparse
from scipy.stats import bootstrap
from scipy.optimize import curve_fit
from tqdm import tqdm
from scipy.linalg import eig

from .schwingerModel import schwingerModel
from . import buildOps as ops
from . import analysis

#calculate the cross correlation between two operators
#there can be different amounts of smearing for the two operators
#this will allow for GEVP analysis
#NOT the most efficient way - written this way to be closest to equations
def getCorrelation(modelObj: schwingerModel, gaugleLinkIndex,
                   smearingOp1, smearingOp2,
                    Gamma1=np.array([[1j,0],[0,-1j]]), Gamma2=np.array([[1j,0],[0,-1j]]),
                    momk=0):

    gaugeLinks = modelObj.linkHistory[gaugleLinkIndex]

    #if the object does NOT have stored props
    #or if that stored prop just happens to be zero
    if(hasattr(modelObj, "storedProps")):
        if((modelObj.storedProps[gaugleLinkIndex] is None)):

            #get total propagator
            dOp = ops.buildDiracOp(modelObj, gaugeLinks)
            prop = np.linalg.inv(dOp.toarray())

            modelObj.storedProps[gaugleLinkIndex] = prop
        else:
            prop = modelObj.storedProps[gaugleLinkIndex]
    else:
        dOp = ops.buildDiracOp(modelObj, gaugeLinks)
        prop = np.linalg.inv(dOp.toarray())

    #build the two smearing operators
    S1 = smearingOp1
    S2 = smearingOp2
    
    #in order to take into account smearing
    #we will have to asymetically smear the propagator
    #See my note on why, but essentially, you can't rotate the same
    #smearing operator in the trace to apply to the same propagator

    prop12 = S1@prop@S2
    prop21 = S2@prop@S1

    stridex = modelObj.dimt*2
    stridet = 2

    correl_conn = np.zeros(modelObj.dimt,dtype=np.complex128)

    #This is the location of the interpolator that is not being
    #we end up just using one edge of this entire correlMat
    t_n = 0
    x_n = 0
    idx_n_start = x_n * stridex + t_n * stridet
    idx_n_end = idx_n_start + 2

    for x_m in range(modelObj.dimx):
        momPhase = np.exp(-1j*2*np.pi*momk*x_m/modelObj.dimx)
        for t_m in range(modelObj.dimt):
            idx_m_start = x_m * stridex + t_m * stridet
            idx_m_end = idx_m_start + 2

            propnm_12 = prop12[idx_n_start:idx_n_end, idx_m_start:idx_m_end]
            propmn_21 = prop21[idx_m_start:idx_m_end, idx_n_start:idx_n_end]

            correl_conn[t_m] += -np.trace(Gamma1@propnm_12@Gamma2@propmn_21)*momPhase

    return np.real(correl_conn)

def correlStats(modelObj: schwingerModel, burnIn=1, autocorrSkip=1,
                    Gamma1=np.array([[1j,0],[0,-1j]]), Gamma2=np.array([[1j,0],[0,-1j]]),
                    kappa1=.1, smearN1=1, kappa2=.1, smearN2=2,
                    momk=0):
    
    acceptedCorrel_conn = []

    #weights for chemicalPot
    weightsMu = analysis.getWeightingFactors(modelObj, 0, burnIn,  autocorrSkip)

    #if k!=0, then each config will show up twice, so we need to repeat the weights
    if(momk!=0):
        weightsMu = np.repeat(weightsMu,2)

    for i in tqdm(range(burnIn,modelObj.metroSteps,autocorrSkip)):
        jacobiS1 = ops.jacobiSmearingOp(modelObj, modelObj.linkHistory[i],kappa1,smearN1)
        jacobiS2 = ops.jacobiSmearingOp(modelObj, modelObj.linkHistory[i],kappa2,smearN2)
        Cconn = getCorrelation(modelObj, i, 
                               smearingOp1=jacobiS1,
                               smearingOp2=jacobiS2,
                               Gamma1=Gamma1, Gamma2=Gamma2, momk=momk)
        
        acceptedCorrel_conn.append(Cconn)

        if(momk!=0):
            Cconn = getCorrelation(modelObj, i, 
                               smearingOp1=jacobiS1,
                               smearingOp2=jacobiS2,
                               Gamma1=Gamma1, Gamma2=Gamma2, momk=-momk)
        
            acceptedCorrel_conn.append(Cconn)


    acceptedCorrel_conn = np.array(acceptedCorrel_conn)

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

def GEVPStats(modelObj: schwingerModel, burnIn=1, autocorrSkip=1,
                    Gamma=np.array([[1j,0],[0,-1j]]),
                    kappa1=.1, smearN1=1, kappa2=0, smearN2=0,
                    momk=0, ti=1):
    
    acceptedCSame1 = []
    acceptedCSame2 = []
    acceptedCMixed = []

    #weights for chemicalPot
    weightsMu = analysis.getWeightingFactors(modelObj, 0, burnIn,  autocorrSkip)

    #if k!=0, then each config will show up twice, so we need to repeat the weights
    if(momk!=0):
        weightsMu = np.repeat(weightsMu,2)

    for i in tqdm(range(burnIn,modelObj.metroSteps,autocorrSkip)):
        jacobiS1 = ops.jacobiSmearingOp(modelObj, modelObj.linkHistory[i],kappa1,smearN1)
        jacobiS2 = ops.jacobiSmearingOp(modelObj, modelObj.linkHistory[i],kappa2,smearN2)

        cSame1 = getCorrelation(modelObj, i, jacobiS1, jacobiS1,
                               Gamma1=Gamma, Gamma2=Gamma, momk=momk)
        cSame2 = getCorrelation(modelObj, i, jacobiS2, jacobiS2,
                               Gamma1=Gamma, Gamma2=Gamma, momk=momk)
        cMixed = getCorrelation(modelObj, i,jacobiS1, jacobiS2,
                               Gamma1=Gamma, Gamma2=Gamma, momk=momk)
        
        acceptedCSame1.append(cSame1)
        acceptedCSame2.append(cSame2)
        acceptedCMixed.append(cMixed)

        if(momk!=0):
            cSame1 = getCorrelation(modelObj, i, jacobiS1, jacobiS1,
                                Gamma1=Gamma, Gamma2=Gamma, momk=-momk)
            cSame2 = getCorrelation(modelObj, i, jacobiS2, jacobiS2,
                                Gamma1=Gamma, Gamma2=Gamma, momk=-momk)
            cMixed = getCorrelation(modelObj, i,jacobiS1, jacobiS2,
                                Gamma1=Gamma, Gamma2=Gamma, momk=-momk)
            
            acceptedCSame1.append(cSame1)
            acceptedCSame2.append(cSame2)
            acceptedCMixed.append(cMixed)


    acceptedCSame1 = np.array(acceptedCSame1)
    acceptedCSame2 = np.array(acceptedCSame2)
    acceptedCMixed = np.array(acceptedCMixed)

    #GEVP all the configs
    #for now we will just keep the lower mass eigenvalues
    #effectively just filters out stuff
    totalCorrels = np.zeros((len(acceptedCSame1),len(acceptedCSame1[0])-ti))
    for i in range(len(acceptedCSame1)):
        newCorrs, basis = gevp(acceptedCSame1[i],acceptedCSame2[i],acceptedCMixed[i],ti=ti)
        totalCorrels[i] = np.real(newCorrs[:,0])

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
    #calculate covariance matrix to allow for fitting
    allcovs = np.array([np.cov(correl_boot[i],rowvar=False) for i in range(len(correl_boot))])
    covMat = np.mean(allcovs, axis=0)

    low  = np.percentile(bootstrap_means, 2.5,  axis=0)
    high = np.percentile(bootstrap_means, 97.5, axis=0)

    totalCorrelMean = np.real(np.average(totalCorrels,axis=0,weights=weightsMu))

    return [totalCorrelMean, np.array([high-totalCorrelMean, totalCorrelMean-low]), covMat]

    
def gevp(corr1, corr2, corrMixed, ti=1):
    correlMat = np.stack([[corr1,corrMixed],[corrMixed,corr2]])

    #this is assuming the the eigenvalues will be sorted in the propoer way
    gevpOutput = [eig(a=correlMat[:,:,i],b=correlMat[:,:,ti]) for i in range(ti, len(corr1))]

    newCorr = np.array([gevpOutput[i][0] for i in range(len(gevpOutput))])
    basis = np.mean(np.array([gevpOutput[i][0] for i in range(len(gevpOutput))]),axis=0)

    return newCorr, basis

def gevpMassExtract(gevpStatsOut,fitT=10):
    def expDecay(nt, Energy):
        return np.exp(-nt*Energy)

    fitMass = curve_fit(expDecay, xdata=np.arange(fitT), 
                    ydata=gevpStatsOut[0][:fitT],sigma=gevpStatsOut[2][:fitT,:fitT],absolute_sigma=True)
    
    return np.array([fitMass[0][0], fitMass[1][0,0]])
