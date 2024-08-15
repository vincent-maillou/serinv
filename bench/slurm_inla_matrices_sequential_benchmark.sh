#!/bin/bash

#SBATCH --job-name=serinv_seq_inlamat
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:2 -C a100_80
#####SBATCH --gres=gpu:a100:1 -C a100_80
#SBATCH --time=00:13:00
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err
#####BATCH --exclusive
#####SBATCH --cpus-per-task=1
#####SBATCH --constraint=a100_80

N_ITERATIONS=5
N_WARMUPS=1


# DIAGONAL_BLOCKSIZE=4002
# ARROWHEAD_BLOCKSIZE=6
# N_DIAG_BLOCKS=250
DIAGONAL_BLOCKSIZE=2865
ARROWHEAD_BLOCKSIZE=4
N_DIAG_BLOCKS=365
# DIAGONAL_BLOCKSIZE=42
# ARROWHEAD_BLOCKSIZE=2
# N_DIAG_BLOCKS=3

FILE_PATH=/home/vault/j101df/j101df10/inla_matrices/INLA_paper_examples/

srun python benchmark_pobtaf_pobtasi.py --diagonal_blocksize $DIAGONAL_BLOCKSIZE --arrowhead_blocksize $ARROWHEAD_BLOCKSIZE --n_diag_blocks $N_DIAG_BLOCKS --file_path $FILE_PATH --n_iterations $N_ITERATIONS --n_warmups $N_WARMUPS
# srun nsys profile -o nsys_serinv_seq_inlamat_n${N_DIAG_BLOCKS}_b${ARROWHEAD_BLOCKSIZE}_a${DIAGONAL_BLOCKSIZE}_%h_%p.qdrep python benchmark_pobtaf_pobtasi.py --diagonal_blocksize $DIAGONAL_BLOCKSIZE --arrowhead_blocksize $ARROWHEAD_BLOCKSIZE --n_diag_blocks $N_DIAG_BLOCKS --file_path $FILE_PATH --n_warmups $N_WARMUPS



