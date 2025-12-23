#!/bin/bash
#SBATCH --job-name=gen_5
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

module load apptainer/1.1.9
module load cuda12.4/toolkit/12.4.1
# Activate your Python environment
source ~/.jackal/bin/activate

# Run your training script
python train.py --config configs/generalization/num_world_5.yaml

echo "Job finished on $(date)"
