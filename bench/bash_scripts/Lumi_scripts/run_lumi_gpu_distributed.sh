#!/bin/bash
####SBATCH --partition=small-g
#SBATCH --partition=standard-g
#SBATCH --account=project_465000929
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=7
#SBATCH --mem-per-gpu=60G
####SBATCH --exclusive
#####SBATCH --output=output-filename

export PATH="/scratch/project_465000929/py312-test/bin:${PATH}"
module load LUMI/23.09 partition/G lumi-CPEtools/1.1-cpeAMD-23.09

export MPICH_GPU_SUPPORT_ENABLED=1
export GMX_FORCE_GPU_AWARE_MPI=1
export GMX_ENABLE_DIRECT_GPU_COMM=1
export MPICH_OFI_NIC_POLICY=GPU
export CUPY_INSTALL_USE_HIP=1
export ROCM_HOME=/opt/rocm
export HCC_AMDGPU_TARGET=gfx90a
export OMP_NUM_THREADS=1
export OMP_PROC_BIND=close
export OMP_PLACES=cores

CPU_BIND="mask_cpu:fe000000000000,fe00000000000000"
CPU_BIND="${CPU_BIND},fe0000,fe000000"
CPU_BIND="${CPU_BIND},fe,fe00"
CPU_BIND="${CPU_BIND},fe00000000,fe0000000000"

file=../../performances/lu_dist/lu_dist_tridiagonal_arrowhead_gpu_performances.py

srun ./select_gpu python ${file} 

## srun -n $SLURM_JOB_NUM_NODES --cpu-bind=${CPU_BIND} ./select_gpu python ${file}

