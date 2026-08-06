import numpy as np
from typing import NamedTuple

_GAMMA_X = np.array([[0, 1], [1, 0]], dtype=complex)
_GAMMA_T = np.array([[0, -1j], [1j, 0]], dtype=complex)
_GAMMA_5 = np.array([[1, 0], [0, -1]], dtype=complex)

_GAMMA_X.flags.writeable = False
_GAMMA_T.flags.writeable = False
_GAMMA_5.flags.writeable = False

class LatticeParams(NamedTuple):
    dimx: int
    dimt: int
    beta: float
    fMass: float
    a: float

    @property
    def gammax(self): return _GAMMA_X

    @property 
    def gammat(self): return _GAMMA_T

class dwfParams(NamedTuple):
    dimx: int
    dimt: int
    dim5: int
    beta: float
    fMass: float
    M5: float
    a: float

    @property
    def gammax(self): return _GAMMA_X

    @property 
    def gammat(self): return _GAMMA_T

    @property
    def gamma5(self): return _GAMMA_5