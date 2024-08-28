#!/bin/bash

#SBATCH --job-name=serinv_generate_dataset
#SBATCH --nodes=1
#SBATCH --gres=gpu:a40:2
#SBATCH --time=00:05:00
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

srun python abg_minibench.py