#!/bin/bash

# #SBATCH --job-name=serinv		   	#Your Job Name
# #SBATCH --nodes=1 			#Number of Nodes desired e.g 1 node
# #SBATCH --gres=gpu:a100:1 			#Run on 1 GPU of any type
# #SBATCH --time=01:01:00 		#Walltime: Duration for the Job to run HH:MM:SS
# #####SBATCH --cpus-per-task=1
# #####SBATCH --constraint=a100_80
# #SBATCH --error=output_serinv.err 		#The .error file name
# #SBATCH --output=output_serinv.out 	#The .output file name
# #SBATCH --exclusive

# conda activate serinv   # activate the conda environment

# DIAGONAL_BLOCKSIZE=4002
# ARROWHEAD_BLOCKSIZE_NSS=0
# ARROWHEAD_BLOCKSIZE_NB=6
# N_DIAG_BLOCKS=250

# DIAGONAL_BLOCKSIZE=42
# ARROWHEAD_BLOCKSIZE_NSS=0
# ARROWHEAD_BLOCKSIZE_NB=2
# N_DIAG_BLOCKS=3

DIAGONAL_BLOCKSIZE=607
ARROWHEAD_BLOCKSIZE_NSS=153
ARROWHEAD_BLOCKSIZE_NB=4
N_DIAG_BLOCKS=1095

#FILE_PATH="/home/x_gaedkelb/serinv/dev/matrices/"
#FILE_PATH=/home/vault/j101df/j101df10/inla_matrices/INLA_paper_examples/
FILE_PATH=/home/vault/j101df/j101df10/inla_matrices/temperature_examples/
DEVICE_STREAMING=True

python benchmark_Chol_selInv.py --diagonal_blocksize $DIAGONAL_BLOCKSIZE --arrowhead_blocksize_nss $ARROWHEAD_BLOCKSIZE_NSS --arrowhead_blocksize_nb $ARROWHEAD_BLOCKSIZE_NB --n_diag_blocks $N_DIAG_BLOCKS --file_path $FILE_PATH --device_streaming $DEVICE_STREAMING
#python benchmark_SchurInv.py --diagonal_blocksize $DIAGONAL_BLOCKSIZE --arrowhead_blocksize $ARROWHEAD_BLOCKSIZE --n_diag_blocks $N_DIAG_BLOCKS --file_path $FILE_PATH --device_streaming $DEVICE_STREAMING

#nsys profile -o nsys_output_serinv_chol_selInv_ns4002_nt250_nb6_%h_%p.txt python benchmark_Chol_selInv.py --diagonal_blocksize $DIAGONAL_BLOCKSIZE --arrowhead_blocksize $ARROWHEAD_BLOCKSIZE --n_diag_blocks $N_DIAG_BLOCKS --file_path $FILE_PATH --device_streaming $DEVICE_STREAMING
#nsys profile -o nsys_output_serinv_SchurInv_ns4002_nt250_nb6_%h_%p.txt python benchmark_SchurInv.py --diagonal_blocksize $DIAGONAL_BLOCKSIZE --arrowhead_blocksize $ARROWHEAD_BLOCKSIZE --n_diag_blocks $N_DIAG_BLOCKS --file_path $FILE_PATH --device_streaming $DEVICE_STREAMING
