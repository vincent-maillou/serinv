#!/bin/bash

DIAGONAL_BLOCKSIZE=92
ARROWHEAD_BLOCKSIZE=4
N_DIAG_BLOCKS=50
FILE_PATH="/home/x_gaedkelb/serinv/dev/matrices/"
DEVICE_STREAMING=True

python benchmark_Chol_selInv.py --diagonal_blocksize $DIAGONAL_BLOCKSIZE --arrowhead_blocksize $ARROWHEAD_BLOCKSIZE --n_diag_blocks $N_DIAG_BLOCKS --file_path $FILE_PATH --device_streaming $DEVICE_STREAMING