#!/bin/bash

#####SBATCH --ntasks-per-node=4
#SBATCH --job-name=testalloc
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:8
#SBATCH --time=00:02:00
#SBATCH --error=output_testalloc.err
#SBATCH --output=output_testalloc.out
#####SBATCH --cpus-per-task=1
#####SBATCH --constraint=a100_80
#####SBATCH --exclusive



# module load intelmpi
# module load mkl
# module load cuda
# module load python/3.9-anaconda
# conda activate serinv

# export CUDAHOME=$CUDA_HOME
# export BASE_PATH=$HOME
# export LD_LIBRARY_PATH=$BASE_PATH/codes/magma-2.7.2/lib:$LD_LIBRARY_PATH
# export CPATH=$BASE_PATH/codes/magma-2.7.2/include:$CPATH
# export LIBRARY_PATH=$BASE_PATH/codes/magma-2.7.2/lib:$LIBRARY_PATH




srun -n 8 --cpu-bind=verbose python testalloc.py


