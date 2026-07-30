from __future__ import annotations

import numpy as np

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .schwingerModel import schwingerModel

def instanton(Q, Lx, Lt):
    x = np.arange(Lx)
    t = np.arange(Lt)
    inst = np.ones((Lx, Lt, 2), dtype=complex)
    # A_x grows linearly in t  ->  constant E field
    inst[:, :, 1] = np.exp(-2j*np.pi*Q*t[None, :]/(Lx*Lt))
    # boundary twist that closes the torus (carries the other 2*pi*Q/Lx per site)
    inst[:, -1, 0] = np.exp(2j*np.pi*Q*x/Lx)
    return inst

def getTopoQ(links):
    Ut = links[:,:,0] # Time links (shape: dimx, dimt)
    Ux = links[:,:,1] # Space links (shape: dimx, dimt)

    # Shift arrays to get U_t(x+1, t) and U_x(x, t+1)
    Ut_shifted_x = np.roll(Ut, shift=-1, axis=0)
    Ux_shifted_t = np.roll(Ux, shift=-1, axis=1)

    # Multiply the four sides of the plaquette
    plaq = Ux * Ut_shifted_x * np.conjugate(Ux_shifted_t) * np.conjugate(Ut)

    Q = np.sum(np.angle(plaq))/(2*np.pi)

    #round to nearest integer
    Q = np.round(Q)

    return Q

def getAllTopoQs(modelObj: schwingerModel):
    Qs = np.zeros(modelObj.metroSteps)

    for i in range(modelObj.metroSteps):
        Qs[i] = getTopoQ(modelObj.linkHistory[i])

    return Qs