from schwingerModel.distillation_gpu import generateDistillFile

#(mass, Nx, numVecs, autocorrSkip) — one entry per cache to fill/build.
#the m=0.01 volume scan, matching the m=0.02 caches volume for volume so the
#two masses stay directly comparable: numVecs 2/2/4/6/8 and 400 groups out to
#Nx=32, with Nx=48/64 held at 100 groups on the wider skip. Everything earlier
#is done: m=0.05, m=0.02 and the m=0.2 Nx=8/16/32 all sit at those counts.
#
#Nx=8/16/32 built on 2026-08-09 morning and are off the list; only the two
#volumes the first sweep never reached are left. The rerun (maxSubSteps=400)
#is producing them now, and runDistill_m0.01_Nx48-64.sh fires this file once
#both ensembles land.
jobs = [
    (0.01, 48, 6, 200),
    (0.01, 64, 8, 200),
]

for m, Nx, numVecs, autocorrSkip in jobs:
    stem = f"./configs/dwf_beta_3.0_m_{m}_Nx_{Nx}_Nt_64_N5_16"
    generateDistillFile(f"{stem}.h5", f"{stem}.hdf5", numVecs=numVecs,
                        autocorrSkip=autocorrSkip, DNums=(0, 2))
