#!/bin/bash -l
#SBATCH --job-name="daint_cpu_sequential"
#SBATCH --account="s1212"
#SBATCH --mail-type=ALL
##SBATCH --mail-user=vmaillou@iis.ee.ethz.ch
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --partition=normal
#SBATCH --constraint=gpu
#SBATCH --hint=nomultithread

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# file=../../performances/lu/cpu_versions/lu_factorize_tridiag_arrowhead_performances.py
file=../../performances/lu/cpu_versions/lu_sinv_tridiag_arrowhead_performances.py

srun python ${file}