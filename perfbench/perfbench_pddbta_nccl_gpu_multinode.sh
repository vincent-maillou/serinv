#!/bin/bash -l
#SBATCH --job-name="perfbench_pddbta_nccl_gpu_multinode"
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err
#SBATCH --account=lp16
###SBATCH --account=sm96
#SBATCH --time=00:08:00
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-task=1
#SBATCH --partition=debug
#### SBATCH --partition=normal
#SBATCH --constraint=gpu
#SBATCH --hint=nomultithread

set -e -u

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MPICH_GPU_SUPPORT_ENABLED=0
# export OMP_PLACES=cores
# export OMP_PROC_BIND=close

export NCCL_NET="AWS Libfabric"

export FI_CXI_RDZV_GET_MIN=0
export FI_CXI_RDZV_THRESHOLD=0
export FI_CXI_RDZV_EAGER_SIZE=0

export NCCL_NET_GDR_LEVEL=PHB
export NCCL_CROSS_NIC=1
export NCCL_PROTO=^LL128
export FI_CXI_DEFAULT_CQ_SIZE=131072
export FI_CXI_DEFAULT_TX_SIZE=16384
export FI_CXI_DISABLE_HOST_REGISTER=1
export FI_CXI_RX_MATCH_MODE=software
export FI_MR_CACHE_MONITOR=userfaultfd

export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID

# export NCCL_DEBUG=INFO

# srun python -c "import cupy as cp; print(cp.show_config())"

srun python ./perfbench_pddbta_nccl_gpu.py --b 1024 --a 256 --n 256 --q 0 --lb 1

