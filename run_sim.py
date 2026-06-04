import numpy as np

from joblib import Parallel, delayed

import pickle

import schwingerModel as sim

m = .2
a = .25
dimx = 64
dimt = 32
beta = 10
totalSteps = 25000

temp = sim.schwingerModel(metroSteps=totalSteps,beta=beta,dimx=dimx,dimt=dimt,aSpacing=a,fMass=m,cgRtol=1e-5)

with open('25kSteps_a_0.25.pkl', 'wb') as f:
    pickle.dump(temp,f)