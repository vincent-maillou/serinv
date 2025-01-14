#!/bin/bash

#SBATCH --job-name=cpu_pobtax
#SBATCH --nodes=1 			#Number of Nodes desired e.g 1 node
#SBATCH --time=00:30:00 		#Walltime: Duration for the Job to run HH:MM:SS
#SBATCH --error=cpu_pobtax.err 		#The .error file name
#SBATCH --output=cpu_pobtax.out 	#The .output file name
#SBATCH --exclusive
#SBATCH --cpus-per-task=72
#SBATCH --partition=spr1tb

unset SLURM_EXPORT_ENV

# set number of threads to requested cpus-per-task
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK 

DIAGONAL_BLOCKSIZE = 2865
ARROWHEAD_BLOCKSIZE = 4
N_DIAG_BLOCKS = 365

N_ITERATIONS = 5
N_WARMUPS = 1

FILE_PATH=/home/vault/j101df/j101df10/inla_matrices/INLA_paper_examples/

python run_cpu_pobtax.py --diagonal_blocksize $DIAGONAL_BLOCKSIZE --arrowhead_blocksize $ARROWHEAD_BLOCKSIZE --n_diag_blocks $N_DIAG_BLOCKS --file_path $FILE_PATH --n_iterations $N_ITERATIONS
