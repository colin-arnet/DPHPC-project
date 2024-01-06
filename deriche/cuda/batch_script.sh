#!/bin/bash -l
#SBATCH --job-name="job_name"
#SBATCH --account="project"
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --constraint=gpu
#SBATCH --hint=nomultithread

srun -w ault06 ./deriche
