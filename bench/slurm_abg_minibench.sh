#!/bin/bash

#SBATCH --job-name=abg_minibench
#SBATCH --nodes=1
#SBATCH --gres=gpu:a40:2
#SBATCH --time=00:05:00
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

srun python abg_minibench.py