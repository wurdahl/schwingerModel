import numpy as np
from joblib import Parallel, delayed
import pickle
import schwingerModel as sim

a = 1
dimx = 8
dimt = 16

R = 10/(32*16) #ratio that we want to keep constant while taking continuum limit

# beta = R*dimx*dimt
beta = 2.0

# m = 0.2 *np.sqrt(10/beta)
m=.2

targetConfigs = 5000
burnIn = 500
nThreads = 10
stepsPerChain = targetConfigs // nThreads

subSteps = 20


def run_chain(seed):
    model = sim.schwingerModel(
        metroSteps=burnIn + stepsPerChain,
        beta=beta, dimx=dimx, dimt=dimt,
        aSpacing=a, fMass=m, cgRtol=1e-5,
        randSeed=seed, tqdmPosition=seed,
        numSubSteps=subSteps
    )
    return model if seed == 0 else model.linkHistory[burnIn:]

if __name__ == '__main__':
    results = Parallel(n_jobs=nThreads)(delayed(run_chain)(seed) for seed in range(nThreads))

    base = results[0]
    merged = np.concatenate([base.linkHistory[burnIn:]] + results[1:])
    base.linkHistory = merged
    base.metroSteps = targetConfigs
    base.storedProps = [None] * targetConfigs

    with open('configs/ryanComp.pkl', 'wb') as f:
        pickle.dump(base, f)
