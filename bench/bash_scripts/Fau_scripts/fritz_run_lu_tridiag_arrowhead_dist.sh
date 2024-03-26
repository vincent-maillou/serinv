#!/bin/bash

#SBATCH --job-name=lu_tridiag_ah_cpu_dist
#SBATCH --partition=multinode
####SBATCH --partition=singlenode
#SBATCH --nodes=2                     #Number of Nodes desired e.g 1 node
#SBATCH --ntasks-per-core=1
####SBATCH --ntasks-per-node=1
####SBATCH --cpus-per-task=64
#SBATCH --hint=nomultithread
#SBATCH --time=00:59:00                 #Walltime: Duration for the Job to run HH:MM:SS
#SBATCH --exclusive
#SBATCH --error=output_%x.err          #The .error file name
#SBATCH --output=output_%x.out         #The .output file name

file=../performances/lu_dist/lu_dist_tridiagonal_arrowhead_performances.py

num_ranks=2
#srun -n ${num_ranks} python ../SDR/bench/performances/lu/cpu_versions/lu_factorize_tridiag_arrowhead_performances.py
echo "srun -n ${num_ranks} python ${file}"
srun -n ${num_ranks} python ${file}
