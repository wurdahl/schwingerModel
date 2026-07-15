import numpy as np
import itertools
from collections import namedtuple

GAMMA5 = np.array([[1j,0],[0,-1j]])

#a meson interpolating operator O = psibar Gamma K psi
#   gamma: (2,2) dirac structure
#   kernel: spatial kernel builder kernel(modelObj, gaugeLinks, nt) -> (dimx,dimx),
#           or None for the identity (e.g. derivativeKernel(n) from distillation.py)
#   momk: integer momentum mode, projects with e^{-i 2 pi momk x / dimx}
mesonOp = namedtuple('mesonOp', ['name', 'gamma', 'kernel', 'momk'], defaults=(0,))

#a single fermion bilinear factor psibar_barFlavor [Gamma (x) e^{-i 2pi momk x/L} K] psi_flavor
#   name keys the elemental cache, so it must uniquely identify (gamma, kernel, momk)
bilinear = namedtuple('bilinear', ['name', 'barFlavor', 'flavor', 'gamma', 'kernel', 'momk'])

#a general interpolating operator: a sum of products of bilinears
#   terms: tuple of (coeff, (bilinear, bilinear, ...))
#single mesons have one bilinear per term, multi-particle operators have several
interpOp = namedtuple('interpOp', ['name', 'terms'])

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


def singleMesonOp(name, flavorTerms=PION_PLUS, gamma=None, kernel=None, momk=0):
    """
    Single-meson interpOp: O = sum_terms coeff * psibar_bf [Gamma e^{-i2pi k x/L} K] psi_f
    All flavor terms share one elemental (keyed by name), so name must be unique
    per (gamma, kernel, momk) within a basis.
    """
    if gamma is None:
        gamma = GAMMA5
    b = lambda fb, f: bilinear(name, fb, f, gamma, kernel, momk)
    terms = tuple((c, (b(fb, f),)) for c, fb, f in flavorTerms)
    return interpOp(name, terms)


def productOp(name, opA, opB):
    """Product of two interpOps, e.g. a two-meson operator (bilinears commute)."""
    terms = tuple((cA*cB, bsA + bsB) for cA, bsA in opA.terms for cB, bsB in opB.terms)
    return interpOp(name, terms)


def opSum(name, coeffsAndOps):
    """Linear combination of interpOps: coeffsAndOps = [(coeff, op), ...]."""
    terms = tuple((c*ct, bs) for c, op in coeffsAndOps for ct, bs in op.terms)
    return interpOp(name, terms)


def piPiOpI1(momk, gamma=None, kernel=None):
    """
    Two-pion operator with single-pion quantum numbers (I=1, I3=+1, P=-1) at zero
    total momentum:
        O = i [ pi+(k) pi0(-k) - pi+(-k) pi0(k) ] / sqrt(2)
    The antisymmetric momentum wavefunction pairs with the antisymmetric (I=1)
    flavor combination, and gives overall parity -1 (two P=-1 pions, odd spatial).
    The overall i makes the cross-correlators with single-pion operators real,
    so the GEVP correlation matrix stays real symmetric.
    """
    pipPos = singleMesonOp(f"pip_k{momk}",  PION_PLUS, gamma, kernel,  momk)
    pipNeg = singleMesonOp(f"pip_k{-momk}", PION_PLUS, gamma, kernel, -momk)
    pi0Pos = singleMesonOp(f"pi0_k{momk}",  PION_ZERO, gamma, kernel,  momk)
    pi0Neg = singleMesonOp(f"pi0_k{-momk}", PION_ZERO, gamma, kernel, -momk)

    return opSum(f"pipi_I1_k{momk}", [( 1j/np.sqrt(2), productOp("_", pipPos, pi0Neg)),
                                      (-1j/np.sqrt(2), productOp("_", pipNeg, pi0Pos))])
