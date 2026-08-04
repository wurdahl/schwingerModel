import numpy as np
from typing import NamedTuple


_GAMMA_X = np.array([[0, 1], [1, 0]], dtype=complex)
_GAMMA_T = np.array([[0, -1j], [1j, 0]], dtype=complex)
_GAMMA_X.flags.writeable = False
_GAMMA_T.flags.writeable = False

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