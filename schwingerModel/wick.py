import numpy as np
import itertools
from collections import namedtuple

#a meson interpolating operator O = psibar Gamma K psi
#   gamma: (2,2) dirac structure
#   kernel: spatial kernel builder kernel(modelObj, gaugeLinks, nt) -> (dimx,dimx),
#           or None for the identity (e.g. derivativeKernel(n) from distillation.py)
mesonOp = namedtuple('mesonOp', ['name', 'gamma', 'kernel'])

#flavor structure of a meson channel: list of (coeff, barFlavor, flavor) terms
#   O = sum_terms coeff * psibar_barFlavor Gamma psi_flavor
PION_PLUS = [(1.0, 'd', 'u')]
PION_ZERO = [(1/np.sqrt(2), 'u', 'u'), (-1/np.sqrt(2), 'd', 'd')]
ETA       = [(1/np.sqrt(2), 'u', 'u'), ( 1/np.sqrt(2), 'd', 'd')]


def permutationCycles(perm):
    """Decompose a permutation (tuple mapping i -> perm[i]) into its cycles."""
    seen = [False]*len(perm)
    cycles = []
    for start in range(len(perm)):
        if seen[start]:
            continue
        cycle = []
        i = start
        while not seen[i]:
            seen[i] = True
            cycle.append(i)
            i = perm[i]
        cycles.append(tuple(cycle))
    return cycles


def wickContractions(bilinears):
    """
    Permutation method for the Wick contractions of a product of fermion bilinears
        < B_0 B_1 ... B_{n-1} >,   B_i = psibar_{f_i} Gamma_i psi_{g_i}

    Every full contraction pairs psi of B_i with psibar of B_{sigma(i)} for some
    permutation sigma, allowed only when the flavors match: g_i == f_{sigma(i)}.
    Each cycle (i1 i2 ... ik) of sigma is a closed fermion loop contributing
        tr[ Gamma_{i1} S(x_{i1},x_{i2}) Gamma_{i2} S(x_{i2},x_{i3}) ... S(x_{ik},x_{i1}) ]
    and the whole contraction carries a sign (-1)^{number of loops}.

    bilinears: list of (barFlavor, flavor) for each B_i
    Returns a list of (sign, cycles) over all allowed permutations.
    """
    n = len(bilinears)
    contractions = []
    for perm in itertools.permutations(range(n)):
        #psi of B_i contracts with psibar of B_{perm[i]}: flavors must match
        if all(bilinears[i][1] == bilinears[perm[i]][0] for i in range(n)):
            cycles = permutationCycles(perm)
            contractions.append(((-1)**len(cycles), cycles))
    return contractions


def twoPointCoeffs(flavorTermsSnk, flavorTermsSrc=None):
    """
    Wick coefficients for the meson two point function < O_snk(t) O_src(0)^dag >.

    Expands both operators over their flavor terms, daggers the source
    ((c psibar_a G psi_b)^dag = c* psibar_b Gbar psi_a), and sums the
    permutation-method contractions by topology.

    Returns (connCoeff, discCoeff):
        C(t) = connCoeff * tr[Phi_snk(t) tau(t,0) PhiBar_src(0) tau(0,t)]
             + discCoeff * tr[Phi_snk(t) tau(t,t)] tr[PhiBar_src(0) tau(0,0)]
    e.g. PION_PLUS/PION_ZERO -> (-1, 0), ETA -> (-1, 2).
    """
    if flavorTermsSrc is None:
        flavorTermsSrc = flavorTermsSnk

    connCoeff = 0.0
    discCoeff = 0.0
    for cSnk, fbSnk, fSnk in flavorTermsSnk:
        for cSrc, fbSrc, fSrc in flavorTermsSrc:
            #daggered source bilinear: psibar and psi flavors swap
            for sign, cycles in wickContractions([(fbSnk, fSnk), (fSrc, fbSrc)]):
                if len(cycles) == 1:
                    connCoeff += cSnk*np.conj(cSrc)*sign
                else:
                    discCoeff += cSnk*np.conj(cSrc)*sign

    return connCoeff, discCoeff
