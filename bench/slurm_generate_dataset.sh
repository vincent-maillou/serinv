#!/bin/bash

#SBATCH --job-name=serinv_generate_dataset
#SBATCH --nodes=1
#SBATCH --gres=gpu:a40:1
#SBATCH --time=00:04:00
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

DIAGONAL_BLOCKSIZE=4096
ARROWHEAD_BLOCKSIZE=0
N_DIAG_BLOCKS=16

N_PROCESSES=1

srun python generate_synthetic_dataset.py --diagonal_blocksize $DIAGONAL_BLOCKSIZE --n_diag_blocks $N_DIAG_BLOCKS --n_processes $N_PROCESSES --arrowhead_blocksize $ARROWHEAD_BLOCKSIZE