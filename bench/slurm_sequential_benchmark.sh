#!/bin/bash

#SBATCH --job-name=serinv_seq_synthetic
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1 -C a100_80
#SBATCH --time=00:15:00
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err
####BATCH --exclusive
#####SBATCH --cpus-per-task=1
#####SBATCH --constraint=a100_80

N_ITERATIONS=10
N_WARMUPS=1

DIAGONAL_BLOCKSIZE=4096
ARROWHEAD_BLOCKSIZE=1024
N_DIAG_BLOCKS=16

# DIAGONAL_BLOCKSIZE=2048
# ARROWHEAD_BLOCKSIZE=512
# N_DIAG_BLOCKS=16

# DIAGONAL_BLOCKSIZE=1024
# ARROWHEAD_BLOCKSIZE=256
# N_DIAG_BLOCKS=16

# DIAGONAL_BLOCKSIZE=512
# ARROWHEAD_BLOCKSIZE=128
# N_DIAG_BLOCKS=16

# DIAGONAL_BLOCKSIZE=256
# ARROWHEAD_BLOCKSIZE=64
# N_DIAG_BLOCKS=16

FILE_PATH=/home/vault/j101df/j101df10/inla_matrices/synthetic_dataset/sequential/

srun python benchmark_pobtaf_pobtasi_synthetic_arrays.py --diagonal_blocksize $DIAGONAL_BLOCKSIZE --arrowhead_blocksize $ARROWHEAD_BLOCKSIZE --n_diag_blocks $N_DIAG_BLOCKS --file_path $FILE_PATH --n_iterations $N_ITERATIONS --n_warmups $N_WARMUPS
# srun nsys profile -o $HOME/codes/serinv/bench/nsys_output_pobtaf_pobtasi_nb${N_DIAG_BLOCKS}_bs${ARROWHEAD_BLOCKSIZE}_as${DIAGONAL_BLOCKSIZE}_%h_%p.qdrep python benchmark_pobtaf_pobtasi_synthetic_arrays.py --diagonal_blocksize $DIAGONAL_BLOCKSIZE --arrowhead_blocksize $ARROWHEAD_BLOCKSIZE --n_diag_blocks $N_DIAG_BLOCKS --file_path $FILE_PATH --n_iterations $N_ITERATIONS --n_warmups $N_WARMUPS
