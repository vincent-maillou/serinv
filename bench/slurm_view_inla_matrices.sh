#!/bin/bash

#SBATCH --job-name=view_inla_matrices
#SBATCH --nodes=1
####SBATCH --gres=gpu:a100:1
#SBATCH --gres=gpu:a40:1
#SBATCH --time=00:01:00
#SBATCH --error=view_inla_matrices.err
#SBATCH --output=view_inla_matrices.out

srun python view_inla_matrices.py