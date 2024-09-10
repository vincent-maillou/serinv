#!/bin/bash

#SBATCH --job-name=serinv_sequential_gpuarray_profile
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1 -C a100_80
#SBATCH --time=00:30:00
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err
####BATCH --exclusive
#####SBATCH --cpus-per-task=1
#####SBATCH --constraint=a100_80

N_ITERATIONS=1
N_WARMUPS=1

DIAGONAL_BLOCKSIZE=2865
ARROWHEAD_BLOCKSIZE=4
N_DIAG_BLOCKS=2


FILE_PATH=/home/vault/j101df/j101df10/inla_matrices/synthetic_dataset/sequential/

# srun python benchmark_gpuarray_pobtaf_pobtasi.py --diagonal_blocksize $DIAGONAL_BLOCKSIZE --arrowhead_blocksize $ARROWHEAD_BLOCKSIZE --n_diag_blocks $N_DIAG_BLOCKS --file_path $FILE_PATH --n_iterations $N_ITERATIONS --n_warmups $N_WARMUPS
# srun nsys profile -o $HOME/codes/serinv/bench/nsys_gpuarray_pobtaf_pobtasi_nb${N_DIAG_BLOCKS}_bs${ARROWHEAD_BLOCKSIZE}_as${DIAGONAL_BLOCKSIZE}_%h_%p.qdrep python benchmark_gpuarray_pobtaf_pobtasi.py --diagonal_blocksize $DIAGONAL_BLOCKSIZE --arrowhead_blocksize $ARROWHEAD_BLOCKSIZE --n_diag_blocks $N_DIAG_BLOCKS --file_path $FILE_PATH --n_iterations $N_ITERATIONS --n_warmups $N_WARMUPS
srun ncu --set full --nvtx -o $HOME/codes/serinv/bench/ncu_gpuarray_pobtaf_pobtasi_nb${N_DIAG_BLOCKS}_bs${ARROWHEAD_BLOCKSIZE}_as${DIAGONAL_BLOCKSIZE}_%h_%p.ncu-rep python benchmark_gpuarray_pobtaf_pobtasi.py --diagonal_blocksize $DIAGONAL_BLOCKSIZE --arrowhead_blocksize $ARROWHEAD_BLOCKSIZE --n_diag_blocks $N_DIAG_BLOCKS --file_path $FILE_PATH --n_iterations $N_ITERATIONS --n_warmups $N_WARMUPS
