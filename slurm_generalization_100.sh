#!/bin/bash
#SBATCH --job-name=gen_100
#SBATCH --output=output_%j.txt
#SBATCH --error=error_%j.txt
#SBATCH --partition=gpu-l40s
#SBATCH --nodes=1
#SBATCH --nodelist=g101
#SBATCH --gres=gpu:4
#SBATCH --ntasks=60
#SBATCH --time=1-00:00:00       # 1 day
#SBATCH --mem=200G

echo "Job started on $(date)"
echo "Running on node: $(hostname)"
nvidia-smi

module load cuda12.4/toolkit/12.4.1
module load apptainer/1.1.9

# Run your training script
python train.py --config configs/generalization/num_world_100.yaml

echo "Job finished on $(date)"
