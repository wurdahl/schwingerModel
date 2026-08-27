"""Regenerate the beta=3 Nx=32 N5=16 caches of the m0 series with the mres
datasets (2026-08-25; the originals, 400 groups at skip 50, went to
configs/old_pre_mres/). Same numVecs, skip 25 -> 800 groups like the m=0 and
m=0.05 members of the series. Needed by mfRelationDwf.ipynb, which puts the
pion mass against m_f = m0 + m_res.
"""
import os
from schwingerModel.distillation_gpu import generateDistillFile

for m in (0.01, 0.02, 0.2):
    stem = f"./configs/dwf_beta_3.0_m_{m}_Nx_32_Nt_64_N5_16"
    if not os.path.exists(f"{stem}.h5"):
        print(f"=== m={m}: no ensemble -- skipped", flush=True); continue
    print(f"=== m={m} -> {stem}.hdf5", flush=True)
    generateDistillFile(f"{stem}.h5", f"{stem}.hdf5", numVecs=4, autocorrSkip=25, DNums=(0, 2))
