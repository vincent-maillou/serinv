#!/bin/bash

#SBATCH --job-name=serinv_distributed
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:8
#SBATCH --time=00:10:00
#SBATCH --error=output_serinv_distributed.err
#SBATCH --output=output_serinv_distributed.out
####BATCH --exclusive
#####SBATCH --cpus-per-task=1
#####SBATCH --constraint=a100_80

N_ITERATIONS=5
N_WARMUPS=1

# DIAGONAL_BLOCKSIZE=1024
# ARROWHEAD_BLOCKSIZE=256
# N_DIAG_BLOCKS=64
# N_PROCESSES=2

DIAGONAL_BLOCKSIZE=1024
ARROWHEAD_BLOCKSIZE=256
N_DIAG_BLOCKS=256
N_PROCESSES=8

# FILE_PATH=/home/vault/j101df/j101df10/inla_matrices/synthetic_dataset/distributed/

srun -n $N_PROCESSES python benchmark_d_pobtaf_d_pobtasi_synthetic_arrays.py --diagonal_blocksize $DIAGONAL_BLOCKSIZE --arrowhead_blocksize $ARROWHEAD_BLOCKSIZE --n_diag_blocks $N_DIAG_BLOCKS --n_iterations $N_ITERATIONS --n_warmups $N_WARMUPS 
