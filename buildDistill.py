from schwingerModel.distillation_gpu import generateDistillFile

#(mass, Nx, numVecs, autocorrSkip) — one entry per cache to fill/build.
#the whole m=0.01 volume scan finished 2026-08-10; every mass now has its
#Nx=8..64 caches at numVecs 2/2/4/6/8, 400 groups out to Nx=32 and 100 groups
#at Nx=48/64 (m=0.2 stops at Nx=32).
#
#The m=0.05/m=0.02 Nx=48 densification to skip=100 finished 2026-08-10 (both
#now hold 200 groups).
#
#Now the fresh m=0 chiral-point ensembles, on the same layout every other mass
#uses: numVecs 2/2/4/6 and 400 groups out to Nx=32, Nx=48 held at 100 groups on
#the wider skip. Nx=64 is not here — that ensemble gets generated overnight and
#its cache (numVecs 8, skip 200) follows afterwards.
#
#The mass must stay a *float* 0.0: the stem interpolates it directly, so 0.0
#gives the "m_0.0" the ensembles are named with, while an int 0 would silently
#build "m_0" paths that match nothing on disk.
#2026-08-11 overnight densification: bring every Nx=48/64 cache to 400 groups
#(autocorrSkip=50, a strict superset of the existing skip=100/200 indices, so
#each job appends only the missing groups). numVecs keeps the per-volume
#convention (48 -> 6, 64 -> 8), which the meta check enforces. The m=0.0 Nx=64
#job is LAST because its ensemble is generated the same night (scan_Nx_m0.0.sh)
#— if that scan fails, everything before it still completes.
jobs = [
    (0.01, 48, 6, 50),
    (0.02, 48, 6, 50),
    (0.05, 48, 6, 50),
    (0.2,  48, 6, 50),
    (0.0,  48, 6, 50),
    (0.01, 64, 8, 50),
    (0.02, 64, 8, 50),
    (0.05, 64, 8, 50),
    (0.2,  64, 8, 50),
    (0.0,  64, 8, 50),
]

for m, Nx, numVecs, autocorrSkip in jobs:
    stem = f"./configs/dwf_beta_3.0_m_{m}_Nx_{Nx}_Nt_64_N5_16"
    generateDistillFile(f"{stem}.h5", f"{stem}.hdf5", numVecs=numVecs,
                        autocorrSkip=autocorrSkip, DNums=(0, 2))
