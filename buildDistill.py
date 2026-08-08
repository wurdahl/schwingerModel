from schwingerModel.distillation_gpu import generateDistillFile

paths = [f"./configs/dwf_beta_3.0_m_0.02_Nx_{Nx}_Nt_64_N5_16.h5" for Nx in [8,16,32,48,64]]

outPaths = [f"./configs/dwf_beta_3.0_m_0.02_Nx_{Nx}_Nt_64_N5_16.hdf5" for Nx in [8,16,32,48,64]]

numVecs = [2,2,4,6,8]

for i in range(len(paths)):
    generateDistillFile(paths[i],outPaths[i],numVecs=numVecs[i],autocorrSkip=200,
                        DNums=(0,2))