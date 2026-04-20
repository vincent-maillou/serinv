#!/bin/bash -l
#SBATCH --job-name="perfbench_ddbt_cpu"
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err
####SBATCH --account=lp16
#SBATCH --account=sm96
#SBATCH --time=00:08:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-task=1
#SBATCH --partition=debug
####SBATCH --partition=normal
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

srun python ./perfbench_ddbt_cpu.py --b 3408 --n 5 --q 1

