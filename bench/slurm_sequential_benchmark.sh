#!/bin/bash

#SBATCH --job-name=serinv_sequential
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

# DIAGONAL_BLOCKSIZE=1024
# ARROWHEAD_BLOCKSIZE=256
# N_DIAG_BLOCKS=32

# DIAGONAL_BLOCKSIZE=512
# ARROWHEAD_BLOCKSIZE=128
# N_DIAG_BLOCKS=128

# DIAGONAL_BLOCKSIZE=4096
# ARROWHEAD_BLOCKSIZE=1024
# N_DIAG_BLOCKS=32

DIAGONAL_BLOCKSIZE=1024
ARROWHEAD_BLOCKSIZE=256
N_DIAG_BLOCKS=64


FILE_PATH=/home/vault/j101df/j101df10/inla_matrices/synthetic_dataset/sequential/

srun python benchmark_pobtaf_pobtasi_synthetic_arrays.py --diagonal_blocksize $DIAGONAL_BLOCKSIZE --arrowhead_blocksize $ARROWHEAD_BLOCKSIZE --n_diag_blocks $N_DIAG_BLOCKS --file_path $FILE_PATH --n_iterations $N_ITERATIONS --n_warmups $N_WARMUPS
# srun nsys profile -o $HOME/codes/serinv/bench/nsys_output_pobtaf_pobtasi_nb${N_DIAG_BLOCKS}_bs${ARROWHEAD_BLOCKSIZE}_as${DIAGONAL_BLOCKSIZE}_%h_%p.qdrep python benchmark_pobtaf_pobtasi_synthetic_arrays.py --diagonal_blocksize $DIAGONAL_BLOCKSIZE --arrowhead_blocksize $ARROWHEAD_BLOCKSIZE --n_diag_blocks $N_DIAG_BLOCKS --file_path $FILE_PATH --n_iterations $N_ITERATIONS --n_warmups $N_WARMUPS
