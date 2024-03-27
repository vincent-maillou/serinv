#!/bin/bash -l
#SBATCH --job-name="daint_gpu_distributed"
#SBATCH --account="s1212"
#SBATCH --mail-type=ALL
##SBATCH --mail-user=vmaillou@iis.ee.ethz.ch
#SBATCH --time=00:20:00
#SBATCH --nodes=128
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --partition=normal
#SBATCH --constraint=gpu
#SBATCH --hint=nomultithread

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.2/lib64/
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export CRAY_CUDA_MPS=1

file=../../performances/lu_dist/lu_dist_tridiagonal_arrowhead_gpu_performances.py

srun -n $SLURM_JOB_NUM_NODES python ${file}