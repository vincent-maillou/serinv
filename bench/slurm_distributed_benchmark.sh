#!/bin/bash

#SBATCH --job-name=serinv_distributed_gpu_08
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100:8 -C a100_80
####SBATCH --qos=a100multi
####SBATCH --gres=gpu:a100:8
#SBATCH --time=00:06:00
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err
#SBATCH --exclusive
#####SBATCH --cpus-per-task=1
#####SBATCH --constraint=a100_80

unset SLURM_EXPORT_ENV

UCX_HOME=" /apps/SPACK/0.19.1/opt/linux-almalinux8-zen/gcc-8.5.0/ucx-1.13.1-woaymodwh7p66njpgt76d7fyqyv7srl3/"
export PATH="${UCX_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${UCX_HOME}/lib:${LD_LIBRARY_PATH}"
which ucx_info

# set number of threads to requested cpus-per-task
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
# for Slurm version >22.05: cpus-per-task has to be set again for srun
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK

# module load python cuda/12.5.1 openmpi/4.1.2-gcc11.2.0-cuda

N_ITERATIONS=5
N_WARMUPS=2

# DIAGONAL_BLOCKSIZE=1024
# ARROWHEAD_BLOCKSIZE=256
# N_DIAG_BLOCKS=64
# N_PROCESSES=2

# DIAGONAL_BLOCKSIZE=1024
# ARROWHEAD_BLOCKSIZE=256
# N_DIAG_BLOCKS=256
# N_PROCESSES=8

# FILE_PATH=/home/vault/j101df/j101df10/inla_matrices/synthetic_dataset/distributed/


DIAGONAL_BLOCKSIZE=2865
ARROWHEAD_BLOCKSIZE=4
N_DIAG_BLOCKS=365
N_PROCESSES=2

FILE_PATH=/home/vault/j101df/j101df10/inla_matrices/INLA_paper_examples/distributed_datasets/



# ...based on the A100 nodes documentation
# GPU : CPU affinity : mask_cpu 
#  0  :    48-63     : 0xffff000000000000
#  1  :    48-63     : 0xffff000000000000
#  2  :    16-31     : 0xffff0000
#  3  :    16-31     : 0xffff0000
#  4  :    112-127   : 0xffff0000000000000000000000000000
#  5  :    112-127   : 0xffff0000000000000000000000000000
#  6  :    80-95     : 0xffff00000000000000000000
#  7  :    80-95     : 0xffff00000000000000000000


# ...modified cpu_mask to use all NUMA domains
# GPU : CPU affinity : mask_cpu 
#  0  :    32-47     : 0xffff00000000
#  1  :    48-63     : 0xffff000000000000
#  2  :    0-15      : 0xffff
#  3  :    16-31     : 0xffff0000
#  4  :    96-111    : 0xffff000000000000000000000000
#  5  :    112-127   : 0xffff0000000000000000000000000000
#  6  :    64-79     : 0xffff0000000000000000
#  7  :    80-95     : 0xffff00000000000000000000

CPU_BIND="mask_cpu:0xffff00000000,0xffff000000000000"
CPU_BIND="${CPU_BIND},0xffff,0xffff0000"
CPU_BIND="${CPU_BIND},0xffff000000000000000000000000,0xffff0000000000000000000000000000"
CPU_BIND="${CPU_BIND},0xffff0000000000000000,0xffff00000000000000000000"

# srun --cpu-bind=verbose ./select_gpu python benchmark_d_pobtaf_d_pobtasi_synthetic_arrays.py --diagonal_blocksize $DIAGONAL_BLOCKSIZE --arrowhead_blocksize $ARROWHEAD_BLOCKSIZE --n_diag_blocks $N_DIAG_BLOCKS --n_iterations $N_ITERATIONS --n_warmups $N_WARMUPS --file_path $FILE_PATH

# echo $SLURM_NTASKS_PER_NODE
# #    -x UCX_TLS=rc,sm,cuda_copy,gdr_copy,cuda_ipc \
# mpirun --mca coll_tuned_use_dynamic_rules 1 \
#        --mca coll_tuned_allreduce_algorithm 5 \
#        --mca opal_cuda_verbose 10 \
#        --mca btl_smcuda_use_cuda_ipc 1 \
#        --mca btl_smcuda_cuda_ipc_verbose 100 \
#        --mca pml ucx \
#        -x UCX_TLS=rc,sm,cuda_copy,gdr_copy,cuda_ipc \
#        -x HCOLL_GPU_ENABLE=1 -x HCOLL_ENABLE_NBC=1 \
# python benchmark_d_pobtaf_d_pobtasi_synthetic_arrays.py --diagonal_blocksize $DIAGONAL_BLOCKSIZE --arrowhead_blocksize $ARROWHEAD_BLOCKSIZE --n_diag_blocks $N_DIAG_BLOCKS --n_iterations $N_ITERATIONS --n_warmups $N_WARMUPS --file_path $FILE_PATH

srun --cpu-bind=${CPU_BIND} --mpi=pmi2 ./select_gpu python benchmark_d_pobtaf_d_pobtasi_synthetic_arrays.py --diagonal_blocksize $DIAGONAL_BLOCKSIZE --arrowhead_blocksize $ARROWHEAD_BLOCKSIZE --n_diag_blocks $N_DIAG_BLOCKS --n_iterations $N_ITERATIONS --n_warmups $N_WARMUPS --file_path $FILE_PATH
# srun --cpu-bind=sockets ./select_gpu python benchmark_d_pobtaf_d_pobtasi_synthetic_arrays.py --diagonal_blocksize $DIAGONAL_BLOCKSIZE --arrowhead_blocksize $ARROWHEAD_BLOCKSIZE --n_diag_blocks $N_DIAG_BLOCKS --n_iterations $N_ITERATIONS --n_warmups $N_WARMUPS --file_path $FILE_PATH
# srun --cpu-bind=${CPU_BIND} -n $N_PROCESSES python benchmark_d_pobtaf_d_pobtasi_synthetic_arrays.py --diagonal_blocksize $DIAGONAL_BLOCKSIZE --arrowhead_blocksize $ARROWHEAD_BLOCKSIZE --n_diag_blocks $N_DIAG_BLOCKS --n_iterations $N_ITERATIONS --n_warmups $N_WARMUPS --file_path $FILE_PATH
# srun -n --cpu-bind=verbose $N_PROCESSES python benchmark_d_pobtaf_d_pobtasi_synthetic_arrays.py --diagonal_blocksize $DIAGONAL_BLOCKSIZE --arrowhead_blocksize $ARROWHEAD_BLOCKSIZE --n_diag_blocks $N_DIAG_BLOCKS --n_iterations $N_ITERATIONS --n_warmups $N_WARMUPS 
