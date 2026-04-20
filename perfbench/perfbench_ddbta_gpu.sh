#!/bin/bash -l
#SBATCH --job-name="perfbench_ddbta_gpu"
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err
#SBATCH --account=lp16
### SBATCH --account=sm96
#SBATCH --time=00:08:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=70
#SBATCH --gpus-per-task=1
#### SBATCH --partition=debug
#SBATCH --partition=normal
#SBATCH --constraint=gpu
#SBATCH --hint=nomultithread
#SBATCH --uenv=prgenv-gnu/25.6:v2
#SBATCH --view=modules

set -e -u

module load cuda gcc meson ninja nccl cray-mpich cmake openblas aws-ofi-nccl netlib-scalapack
conda activate dalia_xccl_daint

srun python ./perfbench_ddbta_gpu.py --b 1024 --a 256 --n 1024 --q 1

