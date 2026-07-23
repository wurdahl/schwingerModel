"""
Topological-sector decomposition tools.

The theta-dependence of any observable can be reassembled exactly from
fixed-sector data:
    <O>_theta = sum_Q e^{i theta Q} Z_Q <O>_Q / sum_Q e^{i theta Q} Z_Q
The phases are applied EXACTLY (no sign problem in the sampling); the physics
inputs are the sector observables <O>_Q (from sector-frozen HMC, hmcJax
fixQ=True) and the relative weights Z_Q/Z_0.

Two sources for the weights:
  - logZQuenched: the exact analytic pure-gauge result. In 2D U(1) on a torus
    the gauge-invariant content is V plaquette angles theta_p in (-pi, pi) with
    the single constraint sum theta_p = 2 pi Q, so
        Z_Q = int dk e^{-2 pi i Q k} g(k)^V,  g(k) = int dtheta e^{beta cos th} cos(k th)
    This is the "theory form": no statistical error, but it omits the fermion
    determinant's (mild, at heavy quark mass) topology suppression.
  - weightsFromQSeries: empirical, from the Q histogram of an ordinary
    tunneling ensemble at theta=0 (P(Q) is proportional to Z_Q by definition).
"""

import numpy as np


def configQ(links):
    """
    Geometric topological charge per configuration from complex link arrays of
    shape (..., dimx, dimt, 2) (numpy mirror of hmcJax.geometricQ).
    """
    Ut, Ux = links[..., 0], links[..., 1]
    plaq = Ux * np.roll(Ut, -1, axis=-2) * np.conj(np.roll(Ux, -1, axis=-1)) * np.conj(Ut)
    return np.sum(np.angle(plaq), axis=(-2, -1)) / (2 * np.pi)


def logZQuenched(beta, V, Qmax, nTheta=4096, nK=4001):
    """
    log(Z_Q / Z_0) for Q = 0..Qmax in pure-gauge 2D U(1) (Wilson action,
    geometric charge), from the constrained-plaquette integral. Exact up to
    quadrature error.
    """
    th = np.linspace(-np.pi, np.pi, nTheta)
    w = np.exp(beta * (np.cos(th) - 1.0))          # rescaled: g(0) stays O(2pi)

    # find kmax where the integrand has decayed away
    def logg(k):
        g = np.trapezoid(w[None, :] * np.cos(np.outer(k, th)), th, axis=1)
        out = np.full(len(k), -np.inf)
        pos = g > 0
        out[pos] = np.log(g[pos])
        return out

    kProbe = np.linspace(0, 3.0, 200)
    h = V * (logg(kProbe) - logg(np.array([0.0]))[0])
    kmax = kProbe[np.searchsorted(-h, 60.0)] if np.any(h < -60) else kProbe[-1]

    k = np.linspace(0, kmax, nK)
    hK = V * (logg(k) - logg(np.array([0.0]))[0])
    integ = np.exp(hK)

    Qs = np.arange(Qmax + 1)
    ZQ = np.trapezoid(integ[None, :] * np.cos(2 * np.pi * np.outer(Qs, k)), k, axis=1)
    return np.log(ZQ / ZQ[0])


def weightsFromQSeries(Qvals, Qmax):
    """
    Empirical log(Z_Q/Z_0) for Q = 0..Qmax from the charges of a tunneling
    theta=0 ensemble, symmetrized over +-Q. Entries with no counts are -inf.
    """
    Qr = np.round(Qvals).astype(int)
    counts = np.array([np.sum(np.abs(Qr) == q) / (1 if q == 0 else 2)
                       for q in range(Qmax + 1)], dtype=float)
    with np.errstate(divide='ignore'):
        return np.log(counts / counts[0])


def reconstructTheta(theta, logZrel, CQ):
    """
    Fixed-sector Fourier reconstruction of a theta-vacuum observable.

    theta   : scalar or (nTheta,) array
    logZrel : (Qmax+1,) log(Z_Q/Z_0) for Q = 0..Qmax (symmetrized in Q)
    CQ      : (Qmax+1, ...) sector observables <O>_Q (parity: <O>_{-Q} = <O>_Q)

    Returns <O>_theta with shape (nTheta, ...):
        sum_Q e^{i theta Q} Z_Q C_Q / sum_Q e^{i theta Q} Z_Q
      = [Z_0 C_0 + 2 sum_{Q>0} cos(theta Q) Z_Q C_Q] / [Z_0 + 2 sum cos(theta Q) Z_Q]
    """
    theta = np.atleast_1d(np.asarray(theta, dtype=float))
    Qs = np.arange(len(logZrel))
    Z = np.exp(logZrel)
    mult = np.where(Qs == 0, 1.0, 2.0)
    w = mult[None, :] * Z[None, :] * np.cos(np.outer(theta, Qs))   # (nTheta, nQ)

    CQ = np.asarray(CQ)
    num = np.tensordot(w, CQ, axes=(1, 0))
    den = np.sum(w, axis=1)
    return num / den.reshape((-1,) + (1,) * (CQ.ndim - 1))
