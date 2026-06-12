import numpy as np
from joblib import Parallel, delayed
import pickle
import schwingerModel as sim

m = 0.2
a = 1
dimx = 32
dimt = 64
beta = 10

targetConfigs = 50000
burnIn = 500
nThreads = 16
stepsPerChain = targetConfigs // nThreads

subSteps = 50

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

    with open('configs/50kSteps_scale_2.pkl', 'wb') as f:
        pickle.dump(base, f)
