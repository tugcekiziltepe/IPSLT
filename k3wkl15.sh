#!/bin/bash
#SBATCH --job-name=nlspae-v6
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4       # Request 4 GPUs per node
#SBATCH --ntasks-per-node=4  # Match the number of GPUs
#SBATCH --nodes=1            # Using a single node
#SBATCH --output=k3wkl15.out
#SBATCH --time=07:30:00

#SBATCH --mem=128G  # Request enough RAM
#SBATCH --cpus-per-task=8  # Allocate sufficient CPU cores

module load Anaconda3
eval "$(conda shell.bash hook)"
conda activate nslpae

echo "Checking GPUs..."
nvidia-smi

srun python3 main.py
