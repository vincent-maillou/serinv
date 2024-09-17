#!/bin/bash -l
#SBATCH --job-name=serinv_distributed_test
#SBATCH --partition=debug
#SBATCH --time=00:10:00
#SBATCH --nodes=1
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
export MPICH_GPU_SUPPORT_ENABLED=1

export PYTHONUNBUFFERED=1

srun ./bench/select_gpu python -m pytest -x --with-mpi tests/algs/test_d_pobtasi.py