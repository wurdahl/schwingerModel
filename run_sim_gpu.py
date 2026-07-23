"""
GPU version of run_sim.py: batched HMC via schwingerModel.hmcJax.

Produces the same kind of pickle as run_sim.py (a schwingerModel object with a
merged, chain-major linkHistory) so the whole measurement stack works unchanged.

Runs in float32 by default -- the RTX 5090 executes FP64 at ~1/64 the FP32
rate, and MD precision only affects the acceptance rate (printed below; if it
sags, raise subSteps). For a float64 run: JAX_ENABLE_X64=1 python run_sim_gpu.py
"""
import pickle

import numpy as np
import schwingerModel as sim

a = 1
dimx = 8
dimt = 16

beta = 2.0
m = .2

targetConfigs = 5000
nChains = 500                 # independent Markov chains advanced together
keptPerChain = targetConfigs // nChains
burnIn = 500                  # per chain, in trajectories
subSteps = 20

if __name__ == '__main__':
    res = sim.hmcJax.runEnsemble(dimx=dimx, dimt=dimt, beta=beta, fMass=m,
                                 aSpacing=a, nChains=nChains, nKept=keptPerChain,
                                 thin=1, burnIn=burnIn, subSteps=subSteps,
                                 cgTol=1e-5, seed=0, start='cold')
    print(f"acceptance: {res['acceptance']:.3f}")

    model = sim.hmcJax.toModel(res, beta=beta, fMass=m, aSpacing=a,
                               subSteps=subSteps, cgTol=1e-5)

    with open('configs/ryanCompGPU.pkl', 'wb') as f:
        pickle.dump(model, f)
    print(f"saved {model.metroSteps} configs to configs/ryanCompGPU.pkl")
