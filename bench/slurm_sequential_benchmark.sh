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

N_ITERATIONS=5
N_WARMUPS=3

# DIAGONAL_BLOCKSIZE=2865
# ARROWHEAD_BLOCKSIZE=4
# N_DIAG_BLOCKS=365
# DIAGONAL_BLOCKSIZE=42
# ARROWHEAD_BLOCKSIZE=2
# N_DIAG_BLOCKS=3

# FILE_PATH=/home/vault/j101df/j101df10/inla_matrices/INLA_paper_examples/
# DEVICE_STREAMING=True

# srun python benchmark_pobtaf_pobtasi.py --diagonal_blocksize $DIAGONAL_BLOCKSIZE --arrowhead_blocksize $ARROWHEAD_BLOCKSIZE --n_diag_blocks $N_DIAG_BLOCKS --file_path $FILE_PATH --device_streaming $DEVICE_STREAMING --n_iterations $N_ITERATIONS





# DIAGONAL_BLOCKSIZE=1024
# ARROWHEAD_BLOCKSIZE=256
# N_DIAG_BLOCKS=32

# DIAGONAL_BLOCKSIZE=512
# ARROWHEAD_BLOCKSIZE=128
# N_DIAG_BLOCKS=128

DIAGONAL_BLOCKSIZE=4096
ARROWHEAD_BLOCKSIZE=1024
N_DIAG_BLOCKS=32


FILE_PATH=/home/vault/j101df/j101df10/inla_matrices/synthetic_dataset/
DEVICE_STREAMING=True

srun python benchmark_pobtaf_pobtasi_synthetic_arrays.py --diagonal_blocksize $DIAGONAL_BLOCKSIZE --arrowhead_blocksize $ARROWHEAD_BLOCKSIZE --n_diag_blocks $N_DIAG_BLOCKS --file_path $FILE_PATH --device_streaming $DEVICE_STREAMING --n_iterations $N_ITERATIONS
srun nsys profile -o nsys_output_pobtaf_pobtasi_nb${N_DIAG_BLOCKS}_bs${ARROWHEAD_BLOCKSIZE}_as${DIAGONAL_BLOCKSIZE}_%h_%p.qdrep python benchmark_pobtaf_pobtasi_synthetic_arrays.py --diagonal_blocksize $DIAGONAL_BLOCKSIZE --arrowhead_blocksize $ARROWHEAD_BLOCKSIZE --n_diag_blocks $N_DIAG_BLOCKS --file_path $FILE_PATH --device_streaming $DEVICE_STREAMING --n_iterations $N_ITERATIONS
