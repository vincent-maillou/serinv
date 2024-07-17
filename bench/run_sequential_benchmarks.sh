#!/bin/bash

DIAGONAL_BLOCKSIZE=4002
ARROWHEAD_BLOCKSIZE=6
N_DIAG_BLOCKS=250
# DIAGONAL_BLOCKSIZE=42
# ARROWHEAD_BLOCKSIZE=2
# N_DIAG_BLOCKS=3

N_ITERATIONS=10
N_WARMUPS=1

#FILE_PATH="/home/x_gaedkelb/serinv/dev/matrices/"
FILE_PATH=/home/vault/j101df/j101df10/inla_matrices/INLA_paper_examples/
# FILE_PATH=/home/vault/j101df/j101df10/inla_matrices/synthetic_dataset/
DEVICE_STREAMING=True

python benchmark_pobtaf_pobtasi.py --diagonal_blocksize $DIAGONAL_BLOCKSIZE --arrowhead_blocksize $ARROWHEAD_BLOCKSIZE --n_diag_blocks $N_DIAG_BLOCKS --file_path $FILE_PATH --device_streaming $DEVICE_STREAMING --n_iterations $N_ITERATIONS
#python benchmark_SchurInv.py --diagonal_blocksize $DIAGONAL_BLOCKSIZE --arrowhead_blocksize $ARROWHEAD_BLOCKSIZE --n_diag_blocks $N_DIAG_BLOCKS --file_path $FILE_PATH --device_streaming $DEVICE_STREAMING

# nsys profile -o nsys_output_serinv_chol_selInv_ns4002_nt250_nb6_%h_%p.qdrep python benchmark_pobtaf_pobtasi.py --diagonal_blocksize $DIAGONAL_BLOCKSIZE --arrowhead_blocksize $ARROWHEAD_BLOCKSIZE --n_diag_blocks $N_DIAG_BLOCKS --file_path $FILE_PATH --device_streaming $DEVICE_STREAMING --warmups $N_WARMUPS
#nsys profile -o nsys_output_serinv_SchurInv_ns4002_nt250_nb6_%h_%p.qdrep python benchmark_SchurInv.py --diagonal_blocksize $DIAGONAL_BLOCKSIZE --arrowhead_blocksize $ARROWHEAD_BLOCKSIZE --n_diag_blocks $N_DIAG_BLOCKS --file_path $FILE_PATH --device_streaming $DEVICE_STREAMING --warmups $N_WARMUPS





# DIAGONAL_BLOCKSIZE=1024
# ARROWHEAD_BLOCKSIZE=256
# N_DIAG_BLOCKS=32

# DIAGONAL_BLOCKSIZE=512
# ARROWHEAD_BLOCKSIZE=128
# N_DIAG_BLOCKS=128

# DIAGONAL_BLOCKSIZE=4096
# ARROWHEAD_BLOCKSIZE=1024
# N_DIAG_BLOCKS=32

# FILE_PATH=/home/vault/j101df/j101df10/inla_matrices/synthetic_dataset/
# DEVICE_STREAMING=True

# python benchmark_pobtaf_pobtasi_synthetic_arrays.py --diagonal_blocksize $DIAGONAL_BLOCKSIZE --arrowhead_blocksize $ARROWHEAD_BLOCKSIZE --n_diag_blocks $N_DIAG_BLOCKS --file_path $FILE_PATH --device_streaming $DEVICE_STREAMING --n_iterations $N_ITERATIONS
