#!/bin/bash

# DIAGONAL_BLOCKSIZE=4002
# ARROWHEAD_BLOCKSIZE=6
# N_DIAG_BLOCKS=250
DIAGONAL_BLOCKSIZE=42
ARROWHEAD_BLOCKSIZE=2
N_DIAG_BLOCKS=3

N_ITERATIONS=5
N_WARMUPS=3

#FILE_PATH="/home/x_gaedkelb/serinv/dev/matrices/"
FILE_PATH=/home/vault/j101df/j101df10/inla_matrices/INLA_paper_examples/
DEVICE_STREAMING=True

python benchmark_pobtaf_pobtasi.py --diagonal_blocksize $DIAGONAL_BLOCKSIZE --arrowhead_blocksize $ARROWHEAD_BLOCKSIZE --n_diag_blocks $N_DIAG_BLOCKS --file_path $FILE_PATH --device_streaming $DEVICE_STREAMING --n_iterations $N_ITERATIONS
#python benchmark_SchurInv.py --diagonal_blocksize $DIAGONAL_BLOCKSIZE --arrowhead_blocksize $ARROWHEAD_BLOCKSIZE --n_diag_blocks $N_DIAG_BLOCKS --file_path $FILE_PATH --device_streaming $DEVICE_STREAMING

# nsys profile -o nsys_output_serinv_chol_selInv_ns4002_nt250_nb6_%h_%p.qdrep python benchmark_pobtaf_pobtasi.py --diagonal_blocksize $DIAGONAL_BLOCKSIZE --arrowhead_blocksize $ARROWHEAD_BLOCKSIZE --n_diag_blocks $N_DIAG_BLOCKS --file_path $FILE_PATH --device_streaming $DEVICE_STREAMING --warmups $N_WARMUPS
#nsys profile -o nsys_output_serinv_SchurInv_ns4002_nt250_nb6_%h_%p.qdrep python benchmark_SchurInv.py --diagonal_blocksize $DIAGONAL_BLOCKSIZE --arrowhead_blocksize $ARROWHEAD_BLOCKSIZE --n_diag_blocks $N_DIAG_BLOCKS --file_path $FILE_PATH --device_streaming $DEVICE_STREAMING --warmups $N_WARMUPS
