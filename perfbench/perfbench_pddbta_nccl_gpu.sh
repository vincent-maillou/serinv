#!/bin/bash -l
#SBATCH --job-name="perfbench_pddbta_nccl_gpu"
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err
#SBATCH --account=lp16
###SBATCH --account=sm96
#SBATCH --time=00:5:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=70
#SBATCH --gpus-per-task=1
#SBATCH --partition=debug
#### SBATCH --partition=normal
#SBATCH --constraint=gpu
#SBATCH --hint=nomultithread
#SBATCH --uenv=prgenv-gnu/24.11:v1
#SBATCH --view=modules

set -e -u

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MPICH_GPU_SUPPORT_ENABLED=1
# export OMP_PLACES=cores
# export OMP_PROC_BIND=close

export NCCL_NET='AWS Libfabric'
export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID

source ~/load_modules.sh
conda activate allin

export MPI_CUDA_AWARE=1

srun python ./perfbench_pddbta_nccl_gpu.py --b 2048 --a 256 --n 32 --q 1 --lb 1

