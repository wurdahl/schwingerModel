"""GPU domain-wall perambulators: batched Schur-preconditioned solves with
mixed-precision refinement.

Mirrors distillation.buildDwfPerambulator (same tau shape, same wall routing,
same contraction) but solves all source columns on the device through the fused
CUDA operator kernel. Each column is a D^{-1} solve, done on the 5D even/odd
Schur complement (see hmc_gpu): one variational CGNE on S^dag S over half the
lattice, with ~1/4 the condition number of the unpreconditioned normal
equations. The mixed-precision structure is the standard lattice-QCD "reliable
updates" scheme: complex64 inner solves (fast on consumer cards whose fp64 runs
at 1/64 rate) wrapped in a complex128 outer refinement loop, with a
warm-started complex128 polish at the end, so the result meets cgRtol in full
double precision no matter where the fp32 solves stall.

Deliberately not imported from __init__: importing jax initializes the GPU.
Use:  from schwingerModel import distillation_gpu
"""
from functools import partial

import numpy as np
import h5py
import jax
import jax.numpy as jnp
from tqdm import tqdm

from .params import dwfParams
from .hmc_gpu import applyD_dwf, packParity, schurSolve
from . import topology as top
from .distillation import (FILE_VERSION, _paramsFromAttrs, findPartialEigenBasis,
                           buildElementalSpatial)


@partial(jax.jit, static_argnums=0)
def _schurBatch64(settings, U64, B64, tol):
    """Inner c64 Schur solves of D dx = r for a batch of columns (lockstep CG)."""
    return jax.vmap(lambda b: schurSolve(settings, U64, b, +1, tol)[0])(B64)


@partial(jax.jit, static_argnums=0)
def _schurBatch128(settings, U128, B, X0, tol):
    """c128 polish solves, warm-started from the refined c64 accumulate's odd half
    (the even half is recomputed exactly by the back-substitution)."""
    return jax.vmap(lambda b, x: schurSolve(settings, U128, b, +1, tol,
                                            x0Odd=packParity(settings, x, 1))[0])(B, X0)


@partial(jax.jit, static_argnums=0)
def _residualBatch(settings, U128, B, X):
    """b - D x per column: the D-system residual, the quantity the perambulator
    actually needs accurate (the old normal-equations residual |b - D Ddag y|
    was the same number in exact arithmetic, since x = Ddag y)."""
    return B - jax.vmap(lambda v: applyD_dwf(settings, U128, v))(X)


def _solveColumns(settings, U128, U64, B, cgRtol, innerTol, maxRefine):
    """x = D^{-1} b per column of B (batch, dim5, dimx, dimt, 2), to cgRtol in c128."""
    bNorm = jnp.sqrt(jnp.sum(jnp.abs(B) ** 2, axis=(1, 2, 3, 4)))
    x = jnp.zeros_like(B)

    #c64 refinement passes: each contracts the c128 residual by roughly the
    #attainable fp32 accuracy; stop early once converged or when fp32 stalls
    prev = jnp.inf
    for _ in range(maxRefine):
        r = _residualBatch(settings, U128, B, x)
        worst = float(jnp.max(jnp.sqrt(jnp.sum(jnp.abs(r) ** 2, axis=(1, 2, 3, 4))) / bNorm))
        if worst < cgRtol or worst > 0.5 * prev:
            break
        prev = worst
        dx = _schurBatch64(settings, U64, r.astype(jnp.complex64), innerTol)
        x = x + dx.astype(jnp.complex128)

    #guaranteed finish: warm-started c128 CG takes over whatever is left
    x = _schurBatch128(settings, U128, B, x, cgRtol)

    r = _residualBatch(settings, U128, B, x)
    worst = float(jnp.max(jnp.sqrt(jnp.sum(jnp.abs(r) ** 2, axis=(1, 2, 3, 4))) / bNorm))
    if worst > 10 * cgRtol:
        raise RuntimeError(f"perambulator solve failed to converge: relative residual {worst:.3e}")

    return x


def buildDwfPerambulator(modelSettings: dwfParams, gaugeLinks, eigVecs,
                         cgRtol=1e-10, innerTol=1e-4, colBatch=128, maxRefine=8,
                         progress=False):
    """GPU version of distillation.buildDwfPerambulator: identical tau, solved
    as batched CGNE on the device. eigVecs: (dimt, dimx, numVecs)."""
    N_t, N_x, N_vec = eigVecs.shape
    N5 = modelSettings.dim5

    U128 = jnp.asarray(gaugeLinks, dtype=jnp.complex128)
    U64 = U128.astype(jnp.complex64)

    #flat column list: (t_src, k, s) -> source on the (s==0 ? 0 : N5-1) wall
    cols = [(t, k, s) for t in range(N_t) for k in range(N_vec) for s in range(2)]
    phiSink = np.empty((len(cols), N_x, N_t, 2), dtype=complex)

    chunks = range(0, len(cols), colBatch)
    for start in (tqdm(chunks) if progress else chunks):
        chunk = cols[start:start + colBatch]
        B = np.zeros((len(chunk), N5, N_x, N_t, 2), dtype=complex)
        for j, (t, k, s) in enumerate(chunk):
            wall = 0 if s == 0 else N5 - 1
            B[j, wall, :, t, s] = eigVecs[t, :, k]

        Phi5 = np.asarray(_solveColumns(modelSettings, U128, U64, jnp.asarray(B),
                                        cgRtol, innerTol, maxRefine))
        #sink walls: spin 0 reads wall N5-1, spin 1 reads wall 0
        phiSink[start:start + len(chunk), :, :, 0] = Phi5[:, N5 - 1, :, :, 0]
        phiSink[start:start + len(chunk), :, :, 1] = Phi5[:, 0, :, :, 1]

    #reassemble (col, x, t_sink, spin) -> (t_src: x, t_sink, s_sink, k, s_src)
    tau = np.zeros((N_t, N_t, N_vec, 2, N_vec, 2), dtype=complex)
    phiSink = phiSink.reshape(N_t, N_vec, 2, N_x, N_t, 2)  # (t_src, k, s_src, x, t_sink, s_sink)
    for t_src in range(N_t):
        Phi = phiSink[t_src].transpose(2, 3, 4, 0, 1)      # (x, t_sink, s_sink, k, s_src)
        tau[:, t_src] = np.einsum('tai, atjkd -> tijkd', eigVecs.conj(), Phi, optimize=True)

    return tau


def _generateConfig(modelSettings, gaugeLinks, numVecs, momks, DNums, cgRtol):
    """Per-config data, mirroring distillation._generateConfig: eigVecs and
    elementals on the host, the perambulator on the device."""
    eigVecs = findPartialEigenBasis(modelSettings, gaugeLinks, numVecs)
    data = {"eigVecs": eigVecs,
            "peram": buildDwfPerambulator(modelSettings, gaugeLinks, eigVecs, cgRtol=cgRtol)}
    for k in momks:
        for n in DNums:
            S = buildElementalSpatial(modelSettings, gaugeLinks, eigVecs, DNum=n, momk=k)
            if np.abs(S).max() < 1e-10:
                raise ValueError(f"p{k}_d{n} unsupported by this basis (momentum window)")
            data[f"elem/p{k}_d{n}"] = S
    return data, top.getTopoQ(gaugeLinks)


def generateDistillFile(ensemblePath, filePath, numVecs,
                        burnIn=0, autocorrSkip=1, momks=(0,), DNums=(0,), cgRtol=1e-10):
    """GPU generation stage: same file layout, attrs, consistency checks and
    incremental-skip semantics as distillation.generateDistillFile, so the
    output is interchangeable with the cpu version (DistillWorkspace.load and
    readDistillMeta read either). dwf ensembles only.

    Configs run sequentially — one process owns the device, and the gpu
    perambulator is fast enough that per-config worker parallelism has nothing
    to add (there is no n_jobs argument).
    """
    with h5py.File(ensemblePath, "r") as ens:
        modelSettings = _paramsFromAttrs(ens.attrs)
        if not isinstance(modelSettings, dwfParams):
            raise ValueError("distillation_gpu only implements the dwf action; "
                             "use distillation.generateDistillFile for wilson ensembles")
        links = ens["links"]
        indices = [int(i) for i in np.arange(burnIn, links.shape[0], autocorrSkip)]

        meta = {"dimx": modelSettings.dimx, "dimt": modelSettings.dimt, "a": modelSettings.a,
                "fMass": modelSettings.fMass, "beta": modelSettings.beta,
                "numVecs": numVecs, "version": FILE_VERSION,
                "fermionAction": "dwf",
                "dim5": modelSettings.dim5, "M5": modelSettings.M5,
                #propagated from the ensemble (see saveEnsemble); in the
                #consistency check, so a cache cannot silently mix the two
                "coldStartForce": bool(ens.attrs["coldStartForce"])}

        with h5py.File(filePath, "a") as f:
            for key, val in meta.items():
                if key in f.attrs:
                    if not np.all(f.attrs[key] == val):
                        raise ValueError(f"{filePath} was generated with {key}={f.attrs[key]}, "
                                         f"requested {key}={val}; use a different file")
                else:
                    f.attrs[key] = val
            if "sourceEnsemble" not in f.attrs:
                f.attrs["sourceEnsemble"] = str(ensemblePath)

            todo = [i for i in indices if f"cfg{i:05d}" not in f]
            if not todo:
                return filePath

            for i in tqdm(todo, desc="Generating distill data (gpu)"):
                data, Q = _generateConfig(modelSettings, links[i], numVecs, momks, DNums, cgRtol)
                grp = f.create_group(f"cfg{i:05d}")
                grp.attrs["Q"] = Q
                for key, arr in data.items():
                    grp.create_dataset(key, data=arr)

    return filePath