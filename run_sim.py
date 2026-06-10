import numpy as np
from joblib import Parallel, delayed
import pickle
import schwingerModel as sim

m = .2
a = .25
dimx = round(16/a)  # divide by a in order to get the same volume
dimt = round(32/a)
beta = 10/a**2      # divide by a**2 in order to get the correct cont limit i.e. same charge

targetConfigs = 10000
burnIn = 500
nThreads = 8
stepsPerChain = targetConfigs // nThreads

def run_chain(seed):
    model = sim.schwingerModel(
        metroSteps=burnIn + stepsPerChain,
        beta=beta, dimx=dimx, dimt=dimt,
        aSpacing=a, fMass=m, cgRtol=1e-5,
        randSeed=seed, tqdmPosition=seed
    )
    return model if seed == 0 else model.linkHistory[burnIn:]

if __name__ == '__main__':
    results = Parallel(n_jobs=nThreads)(delayed(run_chain)(seed) for seed in range(nThreads))

    base = results[0]
    merged = np.concatenate([base.linkHistory[burnIn:]] + results[1:])
    base.linkHistory = merged
    base.metroSteps = targetConfigs
    base.storedProps = [None] * targetConfigs

    with open('configs/10kSteps_a_0.25.pkl', 'wb') as f:
        pickle.dump(base, f)
