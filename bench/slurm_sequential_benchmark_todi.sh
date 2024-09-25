#!/bin/bash -l
#SBATCH --job-name=serinv_distributed_gpu_test_with_timers_4096b_01x01
#SBATCH --output=serinv_distributed_gpu_test_with_timers_4096b_01x01.out
#SBATCH --partition=debug
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --account=lp16
#SBATCH --hint=nomultithread
#SBATCH --hint=exclusive

N_ITERATIONS=20
N_WARMUPS=2

# DIAGONAL_BLOCKSIZE=2865
# ARROWHEAD_BLOCKSIZE=4
# N_DIAG_BLOCKS=365

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

FILE_PATH=$SCRATCH/serinv/datasets/synthetic/

srun --cpu-bind=socket python benchmark_pobtaf_pobtasi_with_timers_synthetic_arrays.py --diagonal_blocksize $DIAGONAL_BLOCKSIZE --arrowhead_blocksize $ARROWHEAD_BLOCKSIZE --n_diag_blocks $N_DIAG_BLOCKS --file_path $FILE_PATH --n_iterations $N_ITERATIONS --n_warmups $N_WARMUPS
# srun nsys profile -o $HOME/codes/serinv/bench/nsys_output_pobtaf_pobtasi_nb${N_DIAG_BLOCKS}_bs${ARROWHEAD_BLOCKSIZE}_as${DIAGONAL_BLOCKSIZE}_%h_%p.qdrep python benchmark_pobtaf_pobtasi_synthetic_arrays.py --diagonal_blocksize $DIAGONAL_BLOCKSIZE --arrowhead_blocksize $ARROWHEAD_BLOCKSIZE --n_diag_blocks $N_DIAG_BLOCKS --file_path $FILE_PATH --n_iterations $N_ITERATIONS --n_warmups $N_WARMUPS
