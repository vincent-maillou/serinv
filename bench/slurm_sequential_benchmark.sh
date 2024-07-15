#!/bin/bash

#SBATCH --job-name=serinv_sequential
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --time=00:05:00
#####SBATCH --cpus-per-task=1
#####SBATCH --constraint=a100_80
#SBATCH --error=output_serinv_sequential.err
#SBATCH --output=output_serinv_sequential.out
#####SBATCH --exclusive

DIAGONAL_BLOCKSIZE=2865
ARROWHEAD_BLOCKSIZE=4
N_DIAG_BLOCKS=365
# DIAGONAL_BLOCKSIZE=42
# ARROWHEAD_BLOCKSIZE=2
# N_DIAG_BLOCKS=3

N_ITERATIONS=5
N_WARMUPS=3

FILE_PATH=/home/vault/j101df/j101df10/inla_matrices/INLA_paper_examples/
DEVICE_STREAMING=True

srun python benchmark_pobtaf_pobtasi.py --diagonal_blocksize $DIAGONAL_BLOCKSIZE --arrowhead_blocksize $ARROWHEAD_BLOCKSIZE --n_diag_blocks $N_DIAG_BLOCKS --file_path $FILE_PATH --device_streaming $DEVICE_STREAMING --n_iterations $N_ITERATIONS
