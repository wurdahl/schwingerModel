"""Residual-mass distill caches for the N5 scans at beta=3, Nx=32, Nt=64.

    python buildDistill_mres.py [m0 [N5 ...]]      default: m0=0.05, every N5

Since 2026-08-24 every dwf cache also stores mres/C_PP and mres/C_JP per config
(point-sink <P P> and <J5q P> by dt), so these caches give m_res(N5) directly.
Layout matches the pre-existing beta=3 Nx=32 caches config-for-config:
numVecs 4, autocorrSkip 25 -> 800 groups. Incremental and safe to re-run: an
existing cache is skipped by generateDistillFile, a missing ensemble is
reported and skipped here.

History: the m=0.05 scan (N5 = 2..32, scan_N5_m0.05.sh) was cached 2026-08-24.
Its N5=16 cache from 2026-08-12 predated the mres datasets; since existing
groups are skipped on rerun (datasets cannot be added to existing groups) it
was deleted and regenerated — same numVecs/skip, so the same cache
config-for-config plus mres/, and the Nx/mass-scan notebooks that read it are
unaffected. The m=0 N5=16 cache (2026-08-11) got the same treatment on
2026-08-25, moved to configs/old_pre_mres/ rather than deleted.
"""
import os
import sys

from schwingerModel.distillation_gpu import generateDistillFile

beta, Nx = 3.0, 32
numVecs, autocorrSkip = 4, 25
n5Default = {0.05: (2, 4, 8, 12, 16, 20, 24, 32),
             0.0: (8, 12, 16, 24, 32)}

m = float(sys.argv[1]) if len(sys.argv) > 1 else 0.05
n5List = [int(a) for a in sys.argv[2:]] or n5Default[m]

for N5 in n5List:
    stem = f"./configs/dwf_beta_{beta}_m_{m}_Nx_{Nx}_Nt_64_N5_{N5}"
    if not os.path.exists(f"{stem}.h5"):
        print(f"=== N5={N5}: no ensemble {stem}.h5 -- skipped", flush=True)
        continue
    print(f"=== N5={N5} -> {stem}.hdf5", flush=True)
    generateDistillFile(f"{stem}.h5", f"{stem}.hdf5", numVecs=numVecs,
                        autocorrSkip=autocorrSkip, DNums=(0, 2))
