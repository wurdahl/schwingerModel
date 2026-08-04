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

- [x] Introduce a small immutable `LatticeParams` (dataclass or NamedTuple):
      `dimx, dimt, beta, fMass, a` (+ the fixed `gammax, gammat`). No behavior.
- [x] Change `buildOps.buildDiracOp`, `buildLaplacian`, `applyCovDerivative`,
      etc. to take `(params, gaugeLinks)` instead of `modelObj` — they already
      use nothing else.
- [x] Split HMC out of the class: a sampler function/class that takes
      `LatticeParams` + run settings and *returns* an ensemble; no more
      side-effect chain in a constructor.
- [x] Define an ensemble container to replace the pickled model object:
      links `(nCfg, dimx, dimt, 2)`, acceptHistory, tunnelAcceptance,
      per-config topological charge Q, params, seeds. Decide format (pickle of
      a plain dataclass, npz, or hdf5) — this becomes the on-disk product of
      `run_sim.py`.
- [x] Update `run_sim.py` to the new interfaces (merging logic stays the same;
      it just concatenates ensemble containers instead of mutating chain 0's
      model object).
- [x] ~~Keep a thin backward-compat shim (or a one-off converter script) for the
      existing `configs/*.pkl` files so old ensembles stay usable.~~ Dropped:
      everything gets regenerated once the GPU path works, so old pickles are
      not worth carrying. They stay readable only until `schwingerModel.py` is
      deleted (pickle needs the class to exist).
- [x] Port the remaining `modelObj` consumers to the ensemble: `analysis.py`
      (`plaqStats`, `getNumDensityRhoBar`, plus a stale `modelObj.dimx` in the
      `getCorrelation` momentum phase), `topology.getAllTopoQs`, and the four
      `modelSettings.metroSteps` reads in `reweighting.py`. They all take
      `(params, gaugeConfigs)` or just `gaugeConfigs` now, with the length
      coming from `len(gaugeConfigs)` instead of a stored step count.
- [x] Drop the dead `TYPE_CHECKING` imports of `schwingerModel` from
      `buildOps.py`, `reweighting.py` and `topology.py`. `distillation.py`
      keeps its one — the module is still wall-to-wall `modelObj` annotations
      until Phase 3.
- [x] Delete `schwingerModel.py` and drop it from `__init__.py`. The
      `configs/*.pkl` ensembles went with it (they needed the class to
      unpickle) and have been deleted. `distillation.py` still has a
      `TYPE_CHECKING` import of it — dead but harmless at runtime, removed in
      Phase 3.

### What changed from the original plan

- `LatticeParams` lives in its own leaf module `params.py` (importing it from
  `schwingerModel.py` would have been a circular import). The gammas are NOT
  tuple fields — they are frozen module-level constants exposed as read-only
  properties, so the tuple stays hashable and `==` compares by value. Array
  fields would have made `hash()` raise and `==` ambiguous, which matters for
  Phase 4 (`params` as a JAX static argument).
- Layout is three modules, not one: `params.py` (physics), `hmc.py` (sampler
  functions, all pure w.r.t. the links they are handed), `experiment.py`
  (chain driver + hdf5 IO + acceptance probe).
- `hmcChain` returns `(params, linkHistory, acceptHistory, tunnelAcceptance)`
  rather than an ensemble object; the merge/trim and the container live one
  level up in `experiment.runExperiment`.
- The on-disk product is hdf5, not a pickled dataclass: `links` chunked per
  config `(1, dimx, dimt, 2)`, `acceptHistory`/`tunnelAcceptance` datasets,
  and attrs for `dimx, dimt, beta, fMass, a, tunneling, cgRtol, numSubSteps,
  seeds, version`. `loadEnsemble` rebuilds `LatticeParams` and returns a
  `SimpleNamespace` (matching `readDistillMeta`) — there is no `Ensemble`
  dataclass yet.
- Q is deliberately NOT stored in the ensemble file. It is a pure function of
  the links, so it gets computed at distillation time instead (Phase 2).
- `metroSteps` in the toml is the TOTAL over all chains (was `targetConfigs`;
  the old name still parses). `burnIn` is still per chain and extra on top.
- Writing refuses to clobber an existing ensemble: `run_sim.py` checks the
  output path before tuning substeps, and `runExperiment` checks before
  spending the chains, so an accidental re-run fails in milliseconds instead
  of after hours.
- The analysis/statistics layer takes `(params, gaugeConfigs)` rather than an
  ensemble object, so it works equally on a loaded ensemble, a slice of one, or
  configs from anywhere else. Nothing downstream of `buildOps` needs a
  container type — only `len()` and indexing.

## Phase 2 — Distillation hdf5 without gauge fields — DONE

Done for decoupling, NOT for size — see the measurement below. `links` were
0.2–0.9% of a distillation file; the point is that a distill file is now
self-sufficient for measurement and theta-reweighting.

- [x] Q per config, stored as a group attr `cfgNNNNN.attrs["Q"]` rather than one
      top-level dataset: the incremental-rerun path appends new groups, so a
      dataset ordered like `configIndices` would silently desync. `readDistillMeta`
      assembles `meta.Q` aligned with `meta.configIndices`.
      `getWeightingFactorsTheta(Qs, theta, burnIn, autocorrSkip)` now takes the
      charges directly and needs neither links nor params.
- [x] `links` dropped from v2, `version` bumped via a `FILE_VERSION` constant so
      the writers cannot drift. v1 files still read — `load` ignores their stale
      links dataset — they just cannot rebuild anything either.
- [x] `gammat`/`gammax` attrs no longer written at v2.
- [x] Resolved by dropping the just-in-time path entirely rather than picking (a)
      or (b): `DistillWorkspace.load` sets `gaugeLinks = None` and never reads
      links, so a loaded workspace is a pure cache. Asking for an elemental that
      was not generated raises a KeyError naming what IS stored, instead of
      attempting a rebuild that v2 cannot support. Generation keeps its on-demand
      computation — that is memoization across ops sharing a spatial part.
      Consequence: `momks`/`DNums` at generation time are a hard commitment.
- [x] `readDistillMeta` gains `Q`, `modelSettings` (rebuilt `LatticeParams`) and
      an int `version`.
- [x] `generateDistillFile(ensemblePath, filePath, numVecs, ...)` reads the
      ensemble file itself instead of taking `(params, gaugeConfigs)`. The params
      come from the ensemble's own attrs so they cannot disagree with the links,
      and configs are read one at a time as tasks dispatch — a 50k x 64x64
      ensemble never lands in memory whole. The distill file records
      `sourceEnsemble` for provenance (outside the meta consistency check, so a
      moved ensemble is not an error).

### Storage reality (measured, per config)

| | 16x32, numVecs=5 | 64x64, numVecs=15 |
|---|---|---|
| `peram` | 1600 KB (90%) | 57600 KB (95%) |
| `elem` x9 | 112 KB (6.4%) | 2025 KB (3.3%) |
| `eigVecs` | 40 KB (2.3%) | 960 KB (1.6%) |
| `links` | 16 KB (0.9%) | 128 KB (0.2%) |

The distillation file is ~10x the ensemble it came from (64x64, 50k configs:
6.55 GB ensemble vs 62 GB distilled), so deleting ensembles to save space is
backwards. The only real lever is the perambulator: storing it as complex64
halves the file (measured 2.00x, gzip is useless at 1.01x) and moves C(t) by
8.6e-9 relative — far below statistical error. NOT done; separate call.

Also note the asymmetry: an ensemble is reproducible from `(randSeed, chains,
settings)` recorded in its attrs, a distillation file is not (the `eigsh`
random start). If one of the two must be recomputable rather than stored, it is
the ensemble.

- [ ] Optional: store `peram` as complex64 for the 2x.
- [ ] Optional: `extendDistillFile` to backfill new momk/DNum elementals into
      existing config groups (needs the ensemble). ~15 lines, 51x cheaper than
      regenerating since it skips the Dirac solve. MUST use each group's stored
      `eigVecs` — a fresh `findPartialEigenBasis` gives a different phase, which
      is invisible at `momk = 0` (the elemental is `V^dag V`) and O(1) wrong at
      `momk != 0`.

## Phase 3 — Distillation without a model stub — DONE

Done: every entry point took `(modelObj, configIndex)` and indexed
`modelObj.linkHistory[configIndex]` itself; they now take `(params, gaugeLinks)`
with the caller doing the indexing, so `configIndex` survives only as the group
label in `load`.

- [x] `findPartialEigenBasis`, `buildPerambulator`, `buildElementalSpatial`,
      `buildElemental` → `(params, gaugeLinks, ...)`; `configIndex` dropped from
      all four (it existed only to index `linkHistory`).
- [x] `buildElemental`'s barred gamma reads `params.gammat`, as does `ws.gamma`.
- [x] `DistillWorkspace(params, gaugeLinks, numVecs, chemicalPot=0)`.
      `DistillWorkspace.load(filePath, configIndex)` keeps its signature —
      `GEVP._measureConfigWick` is the only outside caller — but now builds a
      real `LatticeParams` from the file attrs instead of the `SimpleNamespace`
      stub with its fake `linkHistory={i: ...}` dict.
- [x] `_measureConfig` / `_generateConfig` take `(params, gaugeLinks, ...)`;
      `generateDistillFile(params, gaugeConfigs, filePath, ...)` with `indices`
      built from `len(gaugeConfigs)`. Notebooks feed it `loadEnsemble` output
      directly: `e.modelSettings, e.linkHistory`.
- [x] Dropped the `TYPE_CHECKING` import of `schwingerModel` — the last one in
      the package.
- [x] ~~Store `beta` in the file attrs (it's in the meta dict but double-check
      it lands — the dispersion analysis wanted it and had to hardcode).~~
      Verified: it lands. `configs/50kSteps.hdf5` has attrs
      `a, beta, dimt, dimx, fMass, gammat, gammax, numVecs, version`, so the
      `LatticeParams` rebuild in `load` is complete, not partial.

Validated: existing v1 files still load and contract correctly, and a fresh
generate → load round-trip off a `loadEnsemble` ensemble agrees on the two-point
function and the params.

NOTE for any future validation — `tau` is NOT a valid quantity to diff between
two workspaces built for the same config. `eigsh` starts from a random vector,
so each call returns a different phase/sign per eigenvector; the overlaps
`|<a|b>|` are exactly 1 and the projectors match, but the perambulator carries
the basis phase. Closed contractions cancel it, so compare correlators. Fix
`v0` in `findPartialEigenBasis` if bitwise reproducibility is ever wanted.

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

Status: Phases 1, 2 and 3 are all done, so the pure-Python refactor is complete
and Phase 4 is unblocked. The two remaining Phase-2 items are optional and
independent of everything else.

The notebooks are the one untouched surface: they still call the old
object-based APIs, so they need a pass before any of this is exercised end to
end. Existing `configs/*.hdf5` distillation caches were backfilled with per-config
Q from their stored links (one-off script, files stay version 1 — `readDistillMeta`
detects Q by attribute presence, not version), so theta-reweighting works on them
without regeneration. One early file, `50kSteps_scale_0.5.hdf5`, predates the
`beta` attr and so gets `meta.modelSettings = None`; everything else about it,
Q included, is fine.
