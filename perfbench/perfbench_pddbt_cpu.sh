#!/bin/bash -l
#SBATCH --job-name="perfbench_pddbt_cpu"
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=32
#SBATCH --partition=spr1tb
#SBATCH --hint=nomultithread

unset SLURM_EXPORT_ENV


export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
echo $OMP_NUM_THREADS


CPU_BIND="mask_cpu:0xffff00000000,0xffff000000000000"
CPU_BIND="${CPU_BIND},0xffff,0xffff0000"
CPU_BIND="${CPU_BIND},0xffff000000000000000000000000,0xffff0000000000000000000000000000"
CPU_BIND="${CPU_BIND},0xffff0000000000000000,0xffff00000000000000000000"

srun python ./perfbench_pddbt_cpu.py --b 512 --n 32 --q 0 --lb 1

