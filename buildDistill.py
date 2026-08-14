from schwingerModel.distillation_gpu import generateDistillFile

#(beta, mass, Nx, numVecs, autocorrSkip) — one entry per cache to fill/build.
#beta joined the tuple on 2026-08-12: the stem used to hardcode beta_3.0, which
#silently could not address anything outside the beta=3 series. Both beta and
#mass must stay *floats* — the stem interpolates them directly, so 10.0/0.0 give
#the "beta_10.0"/"m_0.0" the ensembles are named with, while ints would build
#"beta_10"/"m_0" paths matching nothing on disk.
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
#The 2026-08-11 overnight densification (every beta=3 Nx=48/64 cache to 400
#groups at autocorrSkip=50) ran to completion; its list was:
#  (0.01,48,6,50) (0.02,48,6,50) (0.05,48,6,50) (0.2,48,6,50) (0.0,48,6,50)
#  (0.01,64,8,50) (0.02,64,8,50) (0.05,64,8,50) (0.2,64,8,50) (0.0,64,8,50)
#
#The beta=10 Nx=32 cache (numVecs 4, skip 25 -> 800 groups, matching the beta=3
#m=0.05 Nx=32 cache config-for-config) was built 2026-08-12 15:56.
#
#The beta=10 Nx=64 cache (numVecs 8, skip 50 -> 1000 groups; skip held fixed on
#the 50k-config ensemble to preserve the decorrelation criterion rather than
#matching beta=3's 400 groups) was built 2026-08-12 18:25.
#
#TIER 1 (2026-08-13): rebuild every beta=3 Nx=8/16 cache at numVecs=8. The old
#numVecs=2 caches gave a 95-99.8% collinear 2-operator pion basis and cannot
#represent any back-to-back operator beyond |k|=1 (the momentum-k elemental
#rank vanishes at |k|=numVecs); only more eigenvectors enlarge the space. At
#Nx=8, numVecs=8 is the COMPLETE Laplacian eigenbasis (distillation is exact
#there); at Nx=16 it is half the space. skip=50 keeps the 400-group layout.
#The displaced numVecs=2 caches were moved to configs/old_numVecs2/ (the meta
#check refuses a numVecs mismatch in an existing file).
jobs = [
    (3.0, 0.0,   8,  8, 50),
    (3.0, 0.0,  16,  8, 50),
    (3.0, 0.01,  8,  8, 50),
    (3.0, 0.01, 16,  8, 50),
    (3.0, 0.02,  8,  8, 50),
    (3.0, 0.02, 16,  8, 50),
    (3.0, 0.05,  8,  8, 50),
    (3.0, 0.05, 16,  8, 50),
    (3.0, 0.2,   8,  8, 50),
    (3.0, 0.2,  16,  8, 50),
]

for beta, m, Nx, numVecs, autocorrSkip in jobs:
    stem = f"./configs/dwf_beta_{beta}_m_{m}_Nx_{Nx}_Nt_64_N5_16"
    generateDistillFile(f"{stem}.h5", f"{stem}.hdf5", numVecs=numVecs,
                        autocorrSkip=autocorrSkip, DNums=(0, 2))
