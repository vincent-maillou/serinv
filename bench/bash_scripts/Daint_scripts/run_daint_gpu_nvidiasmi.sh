#!/bin/bash -l
#SBATCH --job-name="daint_gpu_sequential"
#SBATCH --account="s1212"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=vmaillou@iis.ee.ethz.ch
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=12
#SBATCH --cpus-per-task=1
#SBATCH --partition=normal
#SBATCH --constraint=gpu
#SBATCH --hint=nomultithread

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export CRAY_CUDA_MPS=1

file=../../performances/lu/gpu_versions/lu_factorize_tridiag_arrowhead_gpu_performances.py
# file=../../performances/lu/gpu_versions/lu_sinv_tridiag_arrowhead_gpu_performances.py

srun nvidia-smi