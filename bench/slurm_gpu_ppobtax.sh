#!/bin/bash -l
#SBATCH --job-name=serinv_gpu_short_distr
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err
#SBATCH --partition=debug
#SBATCH --time=00:30:00 #HH:MM:SS
#SBATCH --nodes=1
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=2 #32 MPI ranks per node
#SBATCH --cpus-per-task=64 #8 OMP threads per rank
#SBATCH --account=lp16
#SBATCH --hint=nomultithread
#SBATCH --hint=exclusive
 
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
# export OMP_NUM_THREADS=1
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export MPICH_MALLOC_FALLBACK=1
export FI_MR_CACHE_MONITOR=userfaultfd
export FI_CXI_DISABLE_HOST_REGISTER=1
export FI_CXI_DEFAULT_CQ_SIZE=131072
export FI_CXI_DEFAULT_TX_SIZE=256
# export NCCL_CROSS_NIC=1
# export NCCL_IGNORE_CPU_AFFINITY=1
export NCCL_NET="AWS Libfabric"
export MPICH_GPU_SUPPORT_ENABLED=1
# export NCCL_DEBUG=INFO

export PYTHONUNBUFFERED=1

ulimit -s unlimited

# FILE_PATH=/capstor/store/cscs/userlab/lp16/vmaillou/dataset/

# DIAGONAL_BLOCKSIZE=2865
# ARROWHEAD_BLOCKSIZE=4
# N_DIAG_BLOCKS=365

DENSITY=0
STRATEGY="allgather"

FILE_PATH=/capstor/store/cscs/userlab/lp16/

DIAGONAL_BLOCKSIZE=4002
ARROWHEAD_BLOCKSIZE=6
N_DIAG_BLOCKS=250

N_ITERATIONS=5
N_WARMUPS=1

cd ${HOME}/serinv/bench
srun --cpu-bind=socket ./select_gpu python run_gpu_ppobtax.py --diagonal_blocksize $DIAGONAL_BLOCKSIZE --arrowhead_blocksize $ARROWHEAD_BLOCKSIZE --n_diag_blocks $N_DIAG_BLOCKS --file_path $FILE_PATH --n_iterations $N_ITERATIONS --density $DENSITY --strategy $STRATEGY
