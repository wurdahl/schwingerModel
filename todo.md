# Refactoring roadmap

Goal: untangle the code so gauge generation, storage, and measurement are
independent stages, then port generation to the GPU. Written as a learning
exercise — each phase should be understood and validated before moving on.
(The old auto-generated JAX experiment is archived at the git tag
`archive/GPU-Gauge` for reference/benchmarks; the plan below is a from-scratch
reimplementation, not a restore.)

## Phase 1 — Decouple everything from the `schwingerModel` object

The object currently plays three roles at once: parameter bag, HMC driver
(it runs the whole chain inside `__init__`!), and ensemble container
(`linkHistory`). Downstream code (`buildOps`, `distillation`) only ever reads
the parameter-bag part plus one config of links.

- [ ] Introduce a small immutable `LatticeParams` (dataclass or NamedTuple):
      `dimx, dimt, beta, fMass, a` (+ the fixed `gammax, gammat`). No behavior.
- [ ] Change `buildOps.buildDiracOp`, `buildLaplacian`, `applyCovDerivative`,
      etc. to take `(params, gaugeLinks)` instead of `modelObj` — they already
      use nothing else.
- [ ] Split HMC out of the class: a sampler function/class that takes
      `LatticeParams` + run settings and *returns* an ensemble; no more
      side-effect chain in a constructor.
- [ ] Define an ensemble container to replace the pickled model object:
      links `(nCfg, dimx, dimt, 2)`, acceptHistory, tunnelAcceptance,
      per-config topological charge Q, params, seeds. Decide format (pickle of
      a plain dataclass, npz, or hdf5) — this becomes the on-disk product of
      `run_sim.py`.
- [ ] Update `run_sim.py` to the new interfaces (merging logic stays the same;
      it just concatenates ensemble containers instead of mutating chain 0's
      model object).
- [ ] Keep a thin backward-compat shim (or a one-off converter script) for the
      existing `configs/*.pkl` files so old ensembles stay usable.

## Phase 2 — Distillation hdf5 without gauge fields

`generateDistillFile` currently stores `links` per config, and
`DistillWorkspace.load` rebuilds a fake model (`SimpleNamespace` stub) around
them. Links are only needed at *generation* time (perambulator, covariant
derivatives) — and later for reweighting, which really needs only Q.

- [ ] Compute the topological charge Q per config at generation time and store
      it (per-config attr or one `Q` dataset indexed like the configs).
      Check `reweighting.getWeightingFactorsTheta` and `topology.py` for what
      exactly reweighting consumes, and make that the stored quantity.
- [ ] Drop the `links` dataset from new files; bump the file `version` attr to 2
      and make readers branch on it (old files keep working).
- [ ] Consequence to resolve: the lazy-elemental path in `DistillWorkspace`
      (building a momk/DNum combination that wasn't precomputed) needs links.
      Options: (a) accept that v2 files must precompute every needed elemental
      up front, or (b) keep links optional behind a flag. Decide and document.
- [ ] `readDistillMeta` gains `Q` (and maybe acceptance stats) so notebooks can
      reweight without touching the ensemble pickle at all.

## Phase 3 — Distillation without a model stub

- [ ] With Phase 1 done, change `findPartialEigenBasis`, `buildPerambulator`,
      `buildElementalSpatial`, `DistillWorkspace` to take
      `(params, gaugeLinks)` / `(params, ensemble, configIndex)` directly.
- [ ] `DistillWorkspace.load` then constructs `LatticeParams` from file attrs —
      no `SimpleNamespace` masquerading as a model, no `linkHistory` dict.
- [ ] `generateDistillFile(ensemble, ...)` instead of `(modelObj, ...)`;
      burnIn/autocorrSkip logic reads ensemble length, not `metroSteps`.
- [ ] Store `beta` in the file attrs (it's in the meta dict but double-check it
      lands — the dispersion analysis wanted it and had to hardcode).

## Phase 4 — GPU port of gauge generation (the big one)

Reimplement the HMC kernel in JAX on top of the Phase-1 interfaces, so the
sampler is swappable (CPU NumPy vs GPU JAX) behind the same ensemble product.

- [ ] Matrix-free Dirac operator: `applyD(params, links, v)` with rolls/stencils
      (no sparse matrices, no explicit D·D†). Validate against
      `buildDiracOp(...) @ v` elementwise.
- [ ] CG on the normal operator via `jax.scipy.sparse.linalg.cg` with the
      matvec closure. Decide fixed-iteration vs tolerance-based (batching runs
      every chain to the slowest chain's count).
- [ ] Leapfrog trajectory as a `lax.scan`, jitted; gauge force from the staple
      formula (or check it against autodiff of the action — good learning
      cross-check).
- [ ] Batched chains with `vmap`; per-chain RNG keys.
- [ ] Precision policy: fp32 MD force, fp64 accept/reject (action + momentum
      sums + pseudofermion bilinear). Known from the archived experiments:
      fp64 runs ~1/64 rate on the 5090; fp32 MD only costs acceptance.
- [ ] Tunneling step: log|det D| has no sparse LU on GPU — keep it on CPU
      (scipy splu on the pulled-back config) or use dense slogdet; once per
      trajectory so either is affordable.
- [ ] Validation ladder (write it BEFORE trusting output): identical-seed force
      comparison vs CPU at fp64, ⟨exp(−ΔH)⟩ = 1, plaquette / Q histograms vs
      CPU ensembles at 16×16 and 32×32, per-sector acceptance at am₀ = 0.
- [ ] `run_sim.py` grows a backend switch (same toml, `backend = "gpu"`), or a
      parallel `run_sim_gpu.py` sharing `loadInput`/tuning logic.
- [ ] Benchmark honestly: time inside a real MD loop with drifting fields
      (warm-started CG on a static field converges in ~1 iteration and
      flatters the numbers). Expectation from the archived benchmarks:
      ~2–5× over 16-core CPU at 32×32–64×64, more only with even-odd
      preconditioning / fused stencils.

## Phase 5 (long term) — Domain wall fermions

- [ ] Kaplan / Shamir formulation in 2D → fermions live on a 2+1d
      slab of extent Ls; physical 2D chiral modes bound to the walls.
- [ ] Implement `applyD_dwf(params, links, v)` — the 2D Wilson kernel from
      Phase 4 reused per s-slice with the ±P_L/P_R hopping in s and the
      domain-wall mass M₅; links do not depend on s.
- [ ] Pauli–Villars fields for the bulk-mode subtraction (pseudofermion action
      ratio det D(m)/det D(1)).
- [ ] HMC force through the 5d operator (Phase-4 CG machinery generalizes; cost
      ~Ls× per solve).
- [ ] Measure the residual mass m_res vs Ls to verify chiral symmetry is
      actually improved — this is the payoff plot, and directly relevant to
      the additive-mass-renormalization headaches at bare am₀ = 0.
- [ ] Decide how distillation interfaces: physical 4d(2d)-boundary propagators
      from the 5d solve (the standard q(x) = P_L ψ(s=0) + P_R ψ(s=Ls−1)
      construction).

## Ordering notes

Phases 1→3 are pure-Python refactors with the existing test surface (notebooks
+ `validate` comparisons against current outputs) and should land before any
GPU work. Phase 4 depends only on Phase 1. Phase 5 depends on Phase 4's
matrix-free kernel but not on Phases 2–3.
