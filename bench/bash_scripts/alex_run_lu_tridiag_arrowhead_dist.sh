#!/bin/bash

#SBATCH --job-name=lu_tridiag_ah_gpu_dist         
#SBATCH --nodes=1                      
###SBATCH --gres=gpu:a40:1                   
#SBATCH --gres=gpu:a100:8                   
#SBATCH --time=01:01:00                 #Walltime: Duration for the Job to run HH:MM:SS
####SBATCH --cpus-per-task=1
#SBATCH --constraint=a100_80
#SBATCH --error=output_%x.err        
#SBATCH --output=output_%x.out       
#SBATCH --exclusive

#module load alex
#conda activate sdr_env

num_ranks=2

file=../SDR/bench/performances/lu_dist/lu_dist_tridiagonal_arrowhead_gpu_performances.py


# mpiexec -n ${num_ranks} python ../SDR/tests/lu_factorize_tests/gpu_versions/lu_factorize_tridiag_arrowhead_gpu_test.py
echo "srun -n ${num_ranks} python ${file}"
srun -n ${num_ranks} python ${file}
