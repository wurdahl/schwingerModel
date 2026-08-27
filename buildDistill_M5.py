"""Residual-mass distill caches for the M5 scan (scan_M5_m0.05.sh): beta=3,
m=0.05, Nx=32, Nt=64, N5 in (8, 16), M5 in (0.5, 0.75, 1.25, 1.5, 1.75).
Same layout as buildDistill_mres.py (numVecs 4, autocorrSkip 25 -> 800 groups)
so every point is config-for-config comparable with the M5=1 scan. Incremental
and safe to re-run; a missing ensemble is reported and skipped.
"""
import os

from schwingerModel.distillation_gpu import generateDistillFile

beta, m, Nx = 3.0, 0.05, 32
numVecs, autocorrSkip = 4, 25

m5Lists = {8: (0.5, 0.75, 1.125, 1.25, 1.375, 1.5, 1.75),
           16: (0.5, 0.75, 1.125, 1.25, 1.375, 1.5, 1.75),
           24: (1.125, 1.25, 1.375, 1.5, 1.625)}

for N5, m5List in m5Lists.items():
    for M5 in m5List:
        stem = f"./configs/dwf_beta_{beta}_m_{m}_Nx_{Nx}_Nt_64_N5_{N5}_M5_{M5}"
        if not os.path.exists(f"{stem}.h5"):
            print(f"=== N5={N5} M5={M5}: no ensemble {stem}.h5 -- skipped", flush=True)
            continue
        print(f"=== N5={N5} M5={M5} -> {stem}.hdf5", flush=True)
        generateDistillFile(f"{stem}.h5", f"{stem}.hdf5", numVecs=numVecs,
                            autocorrSkip=autocorrSkip, DNums=(0, 2))
