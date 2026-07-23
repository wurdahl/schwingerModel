"""
Batched HMC for the 2D Schwinger model on GPU via JAX.

Design notes:
- The lattices here are tiny, so single-config parallelism cannot saturate a
  GPU. Instead every array carries a leading chain axis and jax.vmap runs
  hundreds-to-thousands of independent Markov chains at once.
- Gauge links are stored as REAL angles theta of shape (dimx, dimt, 2) with
  U = exp(i*theta): the momentum update is additive and |U| = 1 is automatic.
  Index convention matches schwingerModel: [..., 0] = time link, [..., 1] = space.
- The fermion force comes from automatic differentiation of the surrogate
  action S_g(theta) - Re[X^dag (D D^dag)(theta) X] with X = (D D^dag)^{-1} phi
  held fixed (stop_gradient): its gradient is the exact HMC force, at the cost
  of a single CG solve per force evaluation, warm-started along the trajectory
  (same solve count as the CPU hmcForcingFunction_vec).
- Precision: validate in float64 (jax.config.update("jax_enable_x64", True)
  BEFORE importing this module) and run production in float32 -- consumer GPUs
  (RTX 5090) execute FP64 at ~1/64 the FP32 rate. MD integration error only
  costs acceptance; the Metropolis step corrects it exactly. Monitor the
  acceptance rate when dropping precision.

All conventions are checked against the CPU implementation by validate_hmc_jax.py.
"""

from functools import partial

import numpy as np

import jax
import jax.numpy as jnp
from jax import lax

_GAMMAX = np.array([[0, 1], [1, 0]], dtype=np.complex128)
_GAMMAT = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)


def links(theta):
    """Angles -> U(1) link variables (complex dtype follows theta's precision)."""
    return jnp.exp(1j * theta)


def gaugeAction(theta, beta):
    """Wilson gauge action S = beta * sum(1 - Re U_plaq), matching totalAction."""
    U = links(theta)
    Ut, Ux = U[..., 0], U[..., 1]
    plaq = Ux * jnp.roll(Ut, -1, axis=0) * jnp.conj(jnp.roll(Ux, -1, axis=1)) * jnp.conj(Ut)
    return beta * jnp.sum(1.0 - jnp.real(plaq))


def applyD(theta, psi, fMass, a, dagger=False):
    """
    Matrix-free Wilson-Dirac apply on a spinor field psi (dimx, dimt, 2spin),
    identical to buildOps.buildDiracOp (chemicalPot=0) in the flattened
    (x, t, spin) basis. dagger=True applies D^dag: the +/- projectors swap while
    the link and antiperiodic-boundary structure keeps the same stencil form.
    """
    U = links(theta)
    Ut, Ux = U[..., 0], U[..., 1]
    dimt = psi.shape[1]

    cdt = psi.dtype
    gx = jnp.asarray(_GAMMAX, dtype=cdt)
    gt = jnp.asarray(_GAMMAT, dtype=cdt)
    I2 = jnp.eye(2, dtype=cdt)
    s = -1.0 if dagger else 1.0

    # antiperiodic fermion BC in time: the sign rides on the wrapped neighbor
    rdt = jnp.real(psi).dtype
    sPos = jnp.ones(dimt, dtype=rdt).at[-1].set(-1.0)
    sNeg = jnp.ones(dimt, dtype=rdt).at[0].set(-1.0)

    hop = (
        Ux[..., None] * jnp.einsum('ij,xtj->xti', I2 - s * gx, jnp.roll(psi, -1, axis=0))
        + jnp.conj(jnp.roll(Ux, 1, axis=0))[..., None]
          * jnp.einsum('ij,xtj->xti', I2 + s * gx, jnp.roll(psi, 1, axis=0))
        + (Ut * sPos[None, :])[..., None]
          * jnp.einsum('ij,xtj->xti', I2 - s * gt, jnp.roll(psi, -1, axis=1))
        + (jnp.conj(jnp.roll(Ut, 1, axis=1)) * sNeg[None, :])[..., None]
          * jnp.einsum('ij,xtj->xti', I2 + s * gt, jnp.roll(psi, 1, axis=1))
    )
    return (fMass + 2.0 / a) * psi - hop / (2.0 * a)


def applyA(theta, psi, fMass, a):
    """A = D D^dag (hermitian positive definite), the pseudofermion kernel."""
    return applyD(theta, applyD(theta, psi, fMass, a, dagger=True), fMass, a)


def _forceSolve(theta, phi, x0, beta, fMass, a, cgTol, maxiter):
    """
    One CG solve X = (D D^dag)^{-1} phi (warm-started at x0), then the exact HMC
    force as the gradient of the surrogate action with X held fixed:
        dS_pf/dtheta = -X^dag (dA/dtheta) X = d/dtheta [ -Re X^dag A(theta) X ]
    Returns (force, X); Re<phi, X> is the pseudofermion action, free of charge.
    """
    X, _ = jax.scipy.sparse.linalg.cg(
        lambda v: applyA(theta, v, fMass, a), phi, x0=x0, tol=cgTol, maxiter=maxiter)
    Xb = lax.stop_gradient(X)

    def surrogate(th):
        return gaugeAction(th, beta) - jnp.real(jnp.vdot(Xb, applyA(th, Xb, fMass, a)))

    return jax.grad(surrogate)(theta), X


def hmcStep(theta, key, beta, fMass, a, subSteps, cgTol, maxiter):
    """
    One HMC trajectory for a single chain (vmap over the leading axis for many).
    Mirrors schwingerModel.hmcStep: pseudofermion heat bath, leapfrog with
    numSubSteps position updates, Metropolis accept/reject on dH.
    Returns (theta_new, accepted).
    """
    eps = 1.0 / subSteps
    kChiR, kChiI, kP, kAcc = jax.random.split(key, 4)

    rdt = theta.dtype
    cdt = jnp.result_type(rdt, jnp.complex64)   # f32 -> c64, f64 -> c128

    chi = ((jax.random.normal(kChiR, theta.shape, rdt)
            + 1j * jax.random.normal(kChiI, theta.shape, rdt))
           / np.sqrt(2)).astype(cdt)
    phi = applyD(theta, chi, fMass, a)
    p0 = jax.random.normal(kP, theta.shape, rdt)

    solve = partial(_forceSolve, beta=beta, fMass=fMass, a=a, cgTol=cgTol, maxiter=maxiter)

    F, X = solve(theta, phi, jnp.zeros_like(phi))
    Spf0 = jnp.real(jnp.vdot(phi, X))
    H0 = 0.5 * jnp.sum(p0 * p0) + gaugeAction(theta, beta) + Spf0

    p = p0 - 0.5 * eps * F
    th = theta

    def body(carry, _):
        th, p, X = carry
        th = th + eps * p
        F, X = solve(th, phi, X)
        p = p - eps * F
        return (th, p, X), None

    (th, p, X), _ = lax.scan(body, (th, p, X), None, length=subSteps - 1)
    th = th + eps * p
    F, X = solve(th, phi, X)
    p = p - 0.5 * eps * F

    Spf1 = jnp.real(jnp.vdot(phi, X))
    dH = (0.5 * jnp.sum(p * p) + gaugeAction(th, beta) + Spf1) - H0

    u = jax.random.uniform(kAcc, (), rdt)
    accept = jnp.isfinite(dH) & (u < jnp.exp(-dH))   # nan dH (CG blowup) -> reject
    thetaNew = jnp.where(accept, th, theta)
    # wrap angles after accept/reject (exact symmetry; keeps floats well-conditioned)
    thetaNew = jnp.mod(thetaNew + jnp.pi, 2 * jnp.pi) - jnp.pi
    return thetaNew, accept


def runEnsemble(dimx, dimt, beta=2.0, fMass=0.2, aSpacing=1.0, nChains=256,
                nKept=100, thin=1, burnIn=100, subSteps=25, cgTol=1e-5,
                maxiter=1000, seed=0, start='cold', progress=True):
    """
    Generate a batched ensemble: nChains independent HMC chains advanced together.

    Every sweep advances all chains by one trajectory. After burnIn sweeps, the
    link configuration of every chain is recorded every `thin` sweeps, nKept
    times, giving nChains*nKept configurations total.

    start: 'cold' (all links 1, like the CPU code) or 'hot' (uniform random
    angles). Hot starts give overdispersed initial conditions -- useful for
    topological-sector coverage -- at the cost of longer burn-in.

    Precision follows jax's x64 setting: enable it before import for float64.

    Returns dict:
      links      : (nKept, nChains, dimx, dimt, 2) complex numpy array
      acceptance : mean acceptance over all trajectories
    """
    rdt = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32

    baseKey = jax.random.key(seed)
    chainIdx = jnp.arange(nChains)

    step = partial(hmcStep, beta=beta, fMass=fMass, a=aSpacing,
                   subSteps=subSteps, cgTol=cgTol, maxiter=maxiter)
    vStep = jax.vmap(step)

    def sweep(carry, _):
        th, gStep, accSum = carry
        sk = jax.random.fold_in(baseKey, gStep)
        cks = jax.vmap(lambda c: jax.random.fold_in(sk, c))(chainIdx)
        th, acc = vStep(th, cks)
        return (th, gStep + 1, accSum + jnp.sum(acc)), None

    @partial(jax.jit, static_argnames='n')
    def sweepN(th, gStep, accSum, n):
        (th, gStep, accSum), _ = lax.scan(sweep, (th, gStep, accSum), None, length=n)
        return th, gStep, accSum

    if start == 'hot':
        theta = jax.random.uniform(jax.random.fold_in(baseKey, -1),
                                   (nChains, dimx, dimt, 2), rdt,
                                   minval=-jnp.pi, maxval=jnp.pi)
    else:
        theta = jnp.zeros((nChains, dimx, dimt, 2), rdt)

    gStep = jnp.zeros((), jnp.int32)
    accSum = jnp.zeros((), rdt)

    if burnIn > 0:
        theta, gStep, accSum = sweepN(theta, gStep, accSum, burnIn)

    iterator = range(nKept)
    if progress:
        try:
            from tqdm.auto import tqdm
            iterator = tqdm(iterator, desc=f"GPU HMC ({nChains} chains)")
        except ImportError:
            pass

    history = []
    for _ in iterator:
        theta, gStep, accSum = sweepN(theta, gStep, accSum, thin)
        history.append(np.asarray(links(theta)))

    nTraj = (burnIn + nKept * thin) * nChains
    return {
        'links': np.stack(history),
        'acceptance': float(accSum) / nTraj,
    }


def toModel(result, beta, fMass, aSpacing=1.0, subSteps=25, cgTol=1e-5):
    """
    Wrap a runEnsemble result in a schwingerModel-compatible object (chain-major
    concatenation, like the run_sim.py merge) so the entire measurement stack --
    analysis, distillation, HDF5 generation -- works unchanged. NOTE: with
    multiple chains, autocorrSkip thinning crosses chain boundaries exactly as
    it does for run_sim.py ensembles.
    """
    from .schwingerModel import schwingerModel as SM

    linksHist = result['links']
    nK, nC, dimx, dimt, _ = linksHist.shape
    merged = np.ascontiguousarray(np.swapaxes(linksHist, 0, 1)) \
               .reshape(nK * nC, dimx, dimt, 2).astype(np.complex128)

    m = object.__new__(SM)
    m.gammax = np.array([[0, 1], [1, 0]])
    m.gammat = np.array([[0, -1j], [1j, 0]])
    m.dimx, m.dimt = dimx, dimt
    m.beta, m.fMass, m.a = beta, fMass, aSpacing
    m.cgRtol, m.numSubSteps = cgTol, subSteps
    m.randSeed, m.rng = None, np.random.default_rng(0)
    m.tqdmPosition, m.previous_CG_ans = 0, None
    m.metroSteps = nK * nC
    m.linkHistory = merged
    m.storedProps = [None] * m.metroSteps
    m.gaugeLinks = merged[-1].copy()
    return m
