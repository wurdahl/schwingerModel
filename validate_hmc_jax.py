"""
Validation ladder for schwingerModel.hmcJax against the CPU implementation.

Runs in float64 (works on CPU jax; on GPU it is slow but exact -- use it once
after any change to the GPU code). Checks, in order:
  1. matrix-free applyD vs the sparse buildOps.buildDiracOp matrix
  2. applyD(dagger=True) vs the conjugate transpose of that matrix
  3. gaugeAction vs schwingerModel.totalAction
  4. autodiff force vs the analytic hmcForcingFunction_vec
  5. leapfrog reversibility (integrate forward, flip momentum, integrate back)
  6. smoke test: a small batched ensemble runs, accepts, and its plaquette
     average lands near the CPU value for the same parameters (optional, --ensemble)
"""

import argparse
import sys
from functools import partial
from types import SimpleNamespace

import numpy as np

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

sys.path.insert(0, ".")
from schwingerModel.schwingerModel import schwingerModel as SM
from schwingerModel import buildOps as ops
from schwingerModel import hmcJax

RNG = np.random.default_rng(7)
FAILURES = []


def check(name, err, tol):
    ok = err < tol
    print(f"  {'PASS' if ok else 'FAIL'}  {name}: max err {err:.3e} (tol {tol:.0e})")
    if not ok:
        FAILURES.append(name)


def stubModel(dimx=8, dimt=16, beta=2.0, m=0.2, a=1.0, cgRtol=1e-12):
    return SimpleNamespace(dimx=dimx, dimt=dimt, beta=beta, fMass=m, a=a,
                           cgRtol=cgRtol, previous_CG_ans=None,
                           gammax=np.array([[0, 1], [1, 0]]),
                           gammat=np.array([[0, -1j], [1j, 0]]))


def randomTheta(stub):
    return RNG.uniform(-np.pi, np.pi, (stub.dimx, stub.dimt, 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ensemble", action="store_true",
                    help="also run the statistical plaquette comparison (slower)")
    args = ap.parse_args()

    stub = stubModel()
    theta = randomTheta(stub)
    U = np.exp(1j * theta)

    # --- 1 & 2: Dirac operator ---------------------------------------------
    print("Dirac operator (8x16, random gauge field):")
    D = ops.buildDiracOp(stub, U).toarray()
    psi = RNG.normal(size=(stub.dimx, stub.dimt, 2)) + 1j * RNG.normal(size=(stub.dimx, stub.dimt, 2))

    ours = np.asarray(hmcJax.applyD(jnp.asarray(theta), jnp.asarray(psi), stub.fMass, stub.a))
    check("applyD vs sparse matrix", np.abs(ours.ravel() - D @ psi.ravel()).max(), 1e-12)

    oursDag = np.asarray(hmcJax.applyD(jnp.asarray(theta), jnp.asarray(psi), stub.fMass, stub.a, dagger=True))
    check("applyD(dagger) vs matrix^dag", np.abs(oursDag.ravel() - D.conj().T @ psi.ravel()).max(), 1e-12)

    # --- 3: gauge action ----------------------------------------------------
    print("Gauge action:")
    sCPU = SM.totalAction(stub, U)
    sJAX = float(hmcJax.gaugeAction(jnp.asarray(theta), stub.beta))
    check("gaugeAction vs totalAction", abs(sCPU - sJAX), 1e-10)

    # --- 4: HMC force -------------------------------------------------------
    print("HMC force (gauge + pseudofermion, autodiff vs analytic):")
    chi = (RNG.normal(size=stub.dimx * stub.dimt * 2)
           + 1j * RNG.normal(size=stub.dimx * stub.dimt * 2)) / np.sqrt(2)
    phi = ops.buildDiracOp(stub, U) @ chi

    fCPU = SM.hmcForcingFunction_vec(stub, U, phi)

    phiField = jnp.asarray(phi.reshape(stub.dimx, stub.dimt, 2))
    fJAX, _ = hmcJax._forceSolve(jnp.asarray(theta), phiField, jnp.zeros_like(phiField),
                                 stub.beta, stub.fMass, stub.a, cgTol=1e-13, maxiter=10000)
    check("force vs hmcForcingFunction_vec", np.abs(np.asarray(fJAX) - fCPU).max(), 1e-8)

    # --- 5: reversibility ---------------------------------------------------
    print("Leapfrog reversibility (tight CG, float64):")
    solve = partial(hmcJax._forceSolve, beta=stub.beta, fMass=stub.fMass, a=stub.a,
                    cgTol=1e-13, maxiter=10000)

    def leapfrog(th, p, nSub):
        eps = 1.0 / nSub
        F, X = solve(th, phiField, jnp.zeros_like(phiField))
        p = p - 0.5 * eps * F
        for _ in range(nSub - 1):
            th = th + eps * p
            F, X = solve(th, phiField, X)
            p = p - eps * F
        th = th + eps * p
        F, _ = solve(th, phiField, X)
        p = p - 0.5 * eps * F
        return th, p

    p0 = jnp.asarray(RNG.normal(size=theta.shape))
    th1, p1 = leapfrog(jnp.asarray(theta), p0, 20)
    th2, p2 = leapfrog(th1, -p1, 20)
    check("theta returns to start", float(jnp.abs(th2 - theta).max()), 1e-9)
    check("momentum returns to -p0", float(jnp.abs(p2 + p0).max()), 1e-9)

    # --- 6: batched ensemble smoke test ------------------------------------
    print("Batched ensemble smoke test (16 chains, 8x8):")
    res = hmcJax.runEnsemble(dimx=8, dimt=8, beta=2.0, fMass=0.5, nChains=16,
                             nKept=25, thin=1, burnIn=25, subSteps=20,
                             seed=1, progress=False)
    plaqGPU = np.real(np.einsum('kcxt,kcxt->', res['links'][..., 1] * np.roll(res['links'][..., 0], -1, axis=2),
                                np.conj(np.roll(res['links'][..., 1], -1, axis=3) * res['links'][..., 0]))
                      ) / res['links'][..., 0].size
    print(f"  acceptance {res['acceptance']:.3f}, <plaq> {plaqGPU:.4f}")
    ok = 0.5 < res['acceptance'] <= 1.0 and np.isfinite(plaqGPU)
    print(f"  {'PASS' if ok else 'FAIL'}  acceptance in (0.5, 1] and plaquette finite")
    if not ok:
        FAILURES.append("ensemble smoke test")

    if args.ensemble:
        print("Statistical plaquette comparison vs CPU (takes a minute):")
        from schwingerModel import analysis
        cpu = SM(dimx=8, dimt=8, metroSteps=300, beta=2.0, fMass=0.5,
                 aSpacing=1, cgRtol=1e-5, numSubSteps=20, randSeed=3)
        pC = analysis.plaqStats(cpu, burnIn=50)
        gpu = hmcJax.runEnsemble(dimx=8, dimt=8, beta=2.0, fMass=0.5, nChains=64,
                                 nKept=50, thin=2, burnIn=50, subSteps=20, seed=2,
                                 progress=False)
        pG = np.array([analysis.getPlaqAvg(l) for l in gpu['links'].reshape(-1, 8, 8, 2)])
        pull = abs(pC[0] - pG.mean()) / np.sqrt(pC[1]**2 + pG.std()**2 / len(pG))
        print(f"  CPU <plaq> = {pC[0]:.4f} +/- {pC[1]:.4f}   GPU <plaq> = {pG.mean():.4f} "
              f"+/- {pG.std()/np.sqrt(len(pG)):.4f}   pull = {pull:.2f}")
        ok = pull < 4
        print(f"  {'PASS' if ok else 'FAIL'}  plaquette agreement (pull < 4)")
        if not ok:
            FAILURES.append("plaquette comparison")

    print()
    if FAILURES:
        print(f"FAILED: {FAILURES}")
        sys.exit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
