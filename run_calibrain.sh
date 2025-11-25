#!/bin/bash
#SBATCH --ntasks=14  # number of "tasks" (default: allocates 1 core per task)
#SBATCH -t 0-02:00:00   # time in d-hh:mm:ss
#SBATCH -o slurm.%j.out # file to save job's STDOUT (%j = JobId)
#SBATCH -e slurm.%j.err # file to save job's STDERR (%j = JobId)
#SBATCH --mail-type=ALL # Send an e-mail when a job starts, stops, or fails
#SBATCH --export=NONE   # Purge the job-submitting shell environment
#SBATCH --gres=gpu
#SBATCH -p equipment_typeG

hostname

nvidia-smi

echo $CUDA_VISIBLE_DEVICES
sleep 20

conda activate calibrain
python run_experiments.py


# srun --partition=equipment_typeG --gpus=1 --pty bash