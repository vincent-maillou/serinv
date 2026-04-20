#!/bin/bash -l
#SBATCH --job-name="perfbench_pddbta_nccl_gpu"
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err
#SBATCH --account=lp16
###SBATCH --account=sm96
#SBATCH --time=00:08:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-task=1
#### SBATCH --partition=debug
#SBATCH --partition=normal
#SBATCH --constraint=gpu
#SBATCH --hint=nomultithread
#SBATCH --uenv=prgenv-gnu/25.6:v2
#SBATCH --view=modules

set -e -u

module load cuda gcc meson ninja nccl cray-mpich cmake openblas aws-ofi-nccl netlib-scalapack
conda activate dalia_xccl_daint

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MPICH_GPU_SUPPORT_ENABLED=0
# export OMP_PLACES=cores
# export OMP_PROC_BIND=close

# For single-node GPU-to-GPU communication, disable network plugins and use GPU P2P
export NCCL_P2P=1
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_READ=1

srun python ./perfbench_pddbta_nccl_gpu.py --b 1024 --a 256 --n 64 --q 1 --lb 1

