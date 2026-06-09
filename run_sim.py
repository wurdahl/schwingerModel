import numpy as np

from joblib import Parallel, delayed

import pickle

import schwingerModel as sim

m = .2
a = .25
dimx = round(16/a) #divide by a in order to get the same volume
dimt = round(32/a)
beta = 10/a**2 #divide by a**2 in order to get the correct cont limit i.e. same charge
totalSteps = 1000

temp = sim.schwingerModel(metroSteps=totalSteps,beta=beta,dimx=dimx,dimt=dimt,aSpacing=a,fMass=m,cgRtol=1e-5)

with open('1kSteps_a_0.25.pkl', 'wb') as f:
    pickle.dump(temp,f)