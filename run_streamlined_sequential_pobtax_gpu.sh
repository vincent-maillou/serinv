#!/bin/bash -l
#SBATCH --job-name="serinv_pobtx_benchmark"
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err
#SBATCH --account=lp16
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-task=1
#SBATCH --partition=debug
#SBATCH --constraint=gpu
#SBATCH --hint=nomultithread
#SBATCH --uenv=prgenv-gnu/24.11:v1
#SBATCH --view=modules

set -e -u

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MPICH_GPU_SUPPORT_ENABLED=1
export OMP_PLACES=cores
export OMP_PROC_BIND=close

export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID

# source ~/load_modules.sh
conda activate serinv_env

# Dataset 1: b = 1675, a = 6, n = 128
# Reference timings (to beat!):
#     - pobtaf: 0.38959
#     - pobtas: 0.02415
#     - pobtasi: 0.29593
# export b=1675
# export a=6
# export n=128

# Dataset 2: b = 4002, a = 6, n = 250
# Reference timings (to beat!):
#     - pobtaf: 3.2716     (INLA_BTA CUDA code: 2.713)
#     - pobtas: 0.15397
#     - pobtasi: 5.15729
export b=4002
export a=6
export n=250

# Benchmark the code
srun python ~/sc25_runs/positive_definite/streamlined_sequential_pobtax_gpu.py --b $b --a $a --n $n

# Profile the code
# srun nsys profile --force-overwrite=true -o profile_serinv_pobtax_b${b}_a${a}_n${n} python ~/repositories/serinv/sc25_runs/positive_definite/streamlined_sequential_pobtax_gpu.py --b $b --a $a --n $n --b $b --a $a --n $n