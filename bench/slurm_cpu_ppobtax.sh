#!/bin/bash

#SBATCH --nodes=1             #Number of Nodes desired e.g 1 node
#SBATCH --ntasks-per-node=2
#SBATCH --time=00:30:00         #Walltime: Duration for the Job to run HH:MM:SS
#SBATCH --error=%x.%j.err        #The .error file name
#SBATCH --output=%x.%j.out     #The .output file name
#SBATCH --exclusive
#SBATCH --cpus-per-task=52
#SBATCH --partition=spr2tb

unset SLURM_EXPORT_ENV

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK 
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK
export OMP_PLACES=cores
export OMP_PROC_BIND=true
# export MKL_NUM_THREADS=52

FILE_PATH=/home/vault/j101df/j101df10/inla_matrices/INLA_paper_examples/

DIAGONAL_BLOCKSIZE=2865
ARROWHEAD_BLOCKSIZE=4
N_DIAG_BLOCKS=365

# DIAGONAL_BLOCKSIZE=4002
# ARROWHEAD_BLOCKSIZE=6
# N_DIAG_BLOCKS=250

N_ITERATIONS=5
N_WARMUPS=1

cd ${HOME}/serinv/bench
srun --cpu-bind=socket python run_cpu_ppobtax.py --diagonal_blocksize $DIAGONAL_BLOCKSIZE --arrowhead_blocksize $ARROWHEAD_BLOCKSIZE --n_diag_blocks $N_DIAG_BLOCKS --file_path $FILE_PATH --n_iterations $N_ITERATIONS
