#!/bin/bash -l
#SBATCH --job-name=serinv_distributed_gpu_test
#SBATCH --time=00:10:00
#SBATCH --nodes=4
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --account=lp16
#SBATCH --hint=nomultithread
#SBATCH --hint=exclusive

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MPICH_MALLOC_FALLBACK=1

export FI_MR_CACHE_MONITOR=userfaultfd
export FI_CXI_DISABLE_HOST_REGISTER=1
export FI_CXI_DEFAULT_CQ_SIZE=131072
export FI_CXI_DEFAULT_TX_SIZE=256
export NCCL_CROSS_NIC=1
export NCCL_IGNORE_CPU_AFFINITY=1
export NCCL_NET="AWS Libfabric"
export MPICH_GPU_SUPPORT_ENABLED=0

export PYTHONUNBUFFERED=1

# unset SLURM_EXPORT_ENV

# # set number of threads to requested cpus-per-task
# export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
# # for Slurm version >22.05: cpus-per-task has to be set again for srun
# export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK

N_ITERATIONS=5
N_WARMUPS=2


DIAGONAL_BLOCKSIZE=2865
ARROWHEAD_BLOCKSIZE=4
N_DIAG_BLOCKS=365
# N_PROCESSES=2

FILE_PATH=$SCRATCH/serinv/datasets/synthetic

srun --cpu-bind=socket ./select_gpu python benchmark_d_pobtaf_d_pobtasi_synthetic_arrays.py --diagonal_blocksize $DIAGONAL_BLOCKSIZE --arrowhead_blocksize $ARROWHEAD_BLOCKSIZE --n_diag_blocks $N_DIAG_BLOCKS --n_iterations $N_ITERATIONS --n_warmups $N_WARMUPS --file_path $FILE_PATH