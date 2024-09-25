#!/bin/bash -l
#SBATCH --job-name=serinv_generate_dataset
#SBATCH --partition=debug
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --account=lp16
#SBATCH --hint=nomultithread
#SBATCH --hint=exclusive

# DIAGONAL_BLOCKSIZE=2865
# ARROWHEAD_BLOCKSIZE=4
# N_DIAG_BLOCKS=365

N_PROCESSES=1
N_DIAG_BLOCKS=16



# for N_PROCESSES in 1 2 4 8 16 32 64
# for ARROWHEAD_BLOCKSIZE in 128 256 512 1024
for ARROWHEAD_BLOCKSIZE in 64
do
    DIAGONAL_BLOCKSIZE=$((4*ARROWHEAD_BLOCKSIZE))
    srun python generate_synthetic_dataset.py --diagonal_blocksize $DIAGONAL_BLOCKSIZE --n_diag_blocks $N_DIAG_BLOCKS --n_processes $N_PROCESSES --arrowhead_blocksize $ARROWHEAD_BLOCKSIZE --path $SCRATCH/serinv/datasets
done