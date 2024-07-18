#!/bin/bash

#SBATCH --job-name=serinv_generate_dataset
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --time=00:08:00
#SBATCH --error=output_generate_dataset.err
#SBATCH --output=output_generate_dataset.out

srun python generate_synthetic_dataset.py 