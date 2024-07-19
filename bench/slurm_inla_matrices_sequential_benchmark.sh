#!/bin/bash

#SBATCH --job-name=serinv_inla_matrices_sequential
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --time=00:15:00
#SBATCH --error=output_serinv_sequential.err
#SBATCH --output=output_serinv_sequential.out
####BATCH --exclusive
#####SBATCH --cpus-per-task=1
#####SBATCH --constraint=a100_80

N_ITERATIONS=5
N_WARMUPS=1

DIAGONAL_BLOCKSIZE=4002
ARROWHEAD_BLOCKSIZE=6
N_DIAG_BLOCKS=250
# DIAGONAL_BLOCKSIZE=2865
# ARROWHEAD_BLOCKSIZE=4
# N_DIAG_BLOCKS=365
# DIAGONAL_BLOCKSIZE=42
# ARROWHEAD_BLOCKSIZE=2
# N_DIAG_BLOCKS=3

FILE_PATH=/home/vault/j101df/j101df10/inla_matrices/INLA_paper_examples/
DEVICE_STREAMING=True


srun python benchmark_pobtaf_pobtasi.py --diagonal_blocksize $DIAGONAL_BLOCKSIZE --arrowhead_blocksize $ARROWHEAD_BLOCKSIZE --n_diag_blocks $N_DIAG_BLOCKS --file_path $FILE_PATH --device_streaming $DEVICE_STREAMING --n_iterations $N_ITERATIONS
# srun nsys profile -o nsys_output_pobtaf_pobtasi_nb${N_DIAG_BLOCKS}_bs${ARROWHEAD_BLOCKSIZE}_as${DIAGONAL_BLOCKSIZE}_%h_%p.qdrep python benchmark_pobtaf_pobtasi.py --diagonal_blocksize $DIAGONAL_BLOCKSIZE --arrowhead_blocksize $ARROWHEAD_BLOCKSIZE --n_diag_blocks $N_DIAG_BLOCKS --file_path $FILE_PATH --device_streaming $DEVICE_STREAMING --warmups $N_WARMUPS



