import numpy as np

"""
Analytic (bosonization) predictions for the multi-flavor Schwinger model,
for comparison against the lattice results.

Conventions: the Wilson gauge action here uses beta = 1/(g a)^2, so the
dimensionless coupling per lattice unit is g a = 1/sqrt(beta).

Main results used (2 degenerate flavors, strong-coupling regime m << g):
  - theta enters only through the combination m cos(theta/2): the anomaly lets
    theta be rotated into the mass term, split evenly between the two flavors,
    so the effective mass perturbation of the bosonized theory is m cos(theta/2).
  - pion (isotriplet) mass: M_pi ~ (m^2 mu)^{1/3} with mu^2 = 2 g^2/pi the
    eta (Schwinger boson) mass scale, giving the parameter-free ratio
        M_pi(theta) / M_pi(0) = |cos(theta/2)|^{2/3}
  - sigma/pion ratio: M_sigma(theta) = sqrt(3) M_pi(theta) (WKB in the
    bosonized theory), sigma stable against sigma -> pi pi for all theta.
  - at theta = pi the leading term vanishes; the true gap is exponentially
    small (SU(2)_1 WZW with marginal perturbation), not literally zero, and
    reweighting from theta=0 loses overlap there anyway.

References: Coleman, Ann. Phys. 101 (1976) 239; Smilga, PRD 55 (1997) 443;
Dempsey-Klebanov-Pufu-Zan arXiv:2305.04437; Itou-Matsumoto-Tanizaki
arXiv:2407.11391 and arXiv:2501.18960 (DMRG/Hamiltonian checks of the
cos^{2/3}(theta/2) prediction).
"""

def coupling(modelObj):
    """Gauge coupling in lattice units: g a = 1/sqrt(beta)."""
    return 1/np.sqrt(modelObj.beta)

def massRatio(modelObj):
    """
    Bare m/g of the ensemble. NOTE: Wilson fermions renormalize the quark mass
    additively, so the physical m/g is smaller than this bare value; use it as
    an upper bound when judging how far into the m << g regime the ensemble is.
    """
    return modelObj.fMass*modelObj.a*np.sqrt(modelObj.beta)

def etaMass(modelObj):
    """Schwinger boson (eta) mass scale in lattice units: mu = g sqrt(2/pi) for Nf=2."""
    return coupling(modelObj)*np.sqrt(2/np.pi)

def pionMassThetaRatio(theta):
    """
    Parameter-free bosonization prediction for the pion mass under a theta term
    (2 degenerate flavors, m << g):
        M_pi(theta)/M_pi(0) = |cos(theta/2)|^{2/3}
    Exact at leading order in the mass perturbation; at theta=pi the true gap is
    exponentially small rather than zero. Corrections grow with m/g.
    """
    return np.abs(np.cos(np.asarray(theta)/2))**(2/3)

def pionMassLO(modelObj, theta=0):
    """
    Leading-order SCALING of the pion mass in lattice units,
        a M_pi ~ [ (a m)^2 (a mu) ]^{1/3} |cos(theta/2)|^{2/3},
    using the BARE quark mass and with the O(1) overall coefficient set to 1
    (the literature coefficient depends on the renormalized mass definition).
    Use pionMassThetaRatio for quantitative comparisons; this is only for
    order-of-magnitude checks.
    """
    return (modelObj.fMass*modelObj.a)**(2/3) * etaMass(modelObj)**(1/3) \
        * pionMassThetaRatio(theta)

#sigma/pion mass ratio from WKB in the bosonized theory, theta-independent
SIGMA_OVER_PION = np.sqrt(3)
