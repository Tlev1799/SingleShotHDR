#!/bin/bash

# --- SLURM Settings ---
#SBATCH --job-name=HDR_e2e
#SBATCH --output=slurm_logs/slurm-%j.out
#SBATCH --error=slurm_logs/slurm-%j.out
#SBATCH --partition=part-preempt
#SBATCH --qos=qos-preempt
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=96:00:00

# --- Setup Environment ---
mkdir -p slurm_logs

# Load user settings
source ~/.bashrc
source /home/projects/sipl-prj10826/miniforge3/etc/profile.d/conda.sh
conda activate PyTorch

# --- Run Code ---
echo "Job started on $(hostname)"
python train.py