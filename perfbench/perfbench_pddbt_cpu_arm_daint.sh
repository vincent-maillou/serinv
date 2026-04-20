#!/bin/bash -l
#SBATCH --job-name="perfbench_pddbt_cpu_arm_daint"
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err
####SBATCH --account=lp16
#SBATCH --account=sm96
#SBATCH --time=00:30:00
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-task=1
####SBATCH --partition=debug
#SBATCH --partition=normal
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

srun python ./perfbench_pddbt_cpu.py --b 3408 --n 80 --q 1 --lb 2.4

