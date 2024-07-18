#!/bin/bash

#SBATCH --job-name=testalloc
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:4
#SBATCH --time=00:01:00
#####SBATCH --cpus-per-task=1
#####SBATCH --constraint=a100_80
#####SBATCH --exclusive

srun -n 4 python testalloc.py


