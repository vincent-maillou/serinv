#!/bin/bash -l
#SBATCH --job-name="daint_cpu_distributed"
#SBATCH --account="s1212"
#SBATCH --mail-type=ALL
##SBATCH --mail-user=vmaillou@iis.ee.ethz.ch
#SBATCH --time=00:30:00
#SBATCH --nodes=128
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --partition=normal
#SBATCH --constraint=gpu
#SBATCH --hint=nomultithread

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

file=../../performances/lu_dist/lu_dist_tridiagonal_arrowhead_performances.py

srun -n $SLURM_JOB_NUM_NODES python ${file}