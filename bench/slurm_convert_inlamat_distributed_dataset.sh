#!/bin/bash

#SBATCH --job-name=serinv_convert_inlamat_to_distributed_dataset
#SBATCH --nodes=1
#SBATCH --gres=gpu:a40:1
#SBATCH --time=00:08:00
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

DIAGONAL_BLOCKSIZE=2865
ARROWHEAD_BLOCKSIZE=4
N_DIAG_BLOCKS=365

N_PROCESSES=32

FILE_PATH=/home/vault/j101df/j101df10/inla_matrices/INLA_paper_examples/

srun python convert_inlamat_to_distributed_dataset.py --diagonal_blocksize $DIAGONAL_BLOCKSIZE --n_blocks $N_DIAG_BLOCKS --n_processes $N_PROCESSES --arrowhead_blocksize $ARROWHEAD_BLOCKSIZE --file_path $FILE_PATH