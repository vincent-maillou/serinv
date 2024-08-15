#!/bin/bash

#SBATCH --job-name=serinv_distributed
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:2 -C a100_80
####SBATCH --qos=a100multi
####SBATCH --gres=gpu:a100:8
#SBATCH --time=00:06:00
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err
####BATCH --exclusive
#####SBATCH --cpus-per-task=1
#####SBATCH --constraint=a100_80

N_ITERATIONS=1
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

srun --cpu-bind=${CPU_BIND} -n $N_PROCESSES python benchmark_d_pobtaf_d_pobtasi_synthetic_arrays.py --diagonal_blocksize $DIAGONAL_BLOCKSIZE --arrowhead_blocksize $ARROWHEAD_BLOCKSIZE --n_diag_blocks $N_DIAG_BLOCKS --n_iterations $N_ITERATIONS --n_warmups $N_WARMUPS --file_path $FILE_PATH
# srun -n --cpu-bind=verbose $N_PROCESSES python benchmark_d_pobtaf_d_pobtasi_synthetic_arrays.py --diagonal_blocksize $DIAGONAL_BLOCKSIZE --arrowhead_blocksize $ARROWHEAD_BLOCKSIZE --n_diag_blocks $N_DIAG_BLOCKS --n_iterations $N_ITERATIONS --n_warmups $N_WARMUPS 
