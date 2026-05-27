#!/bin/bash
# ==============================================================================
# Friday, June 5 - Morning Session
# SLURM Batch Job Script for DINOv2 Training Launch
# SOLUTION CODE — instructor reference only, do not share with students.
# ==============================================================================
#SBATCH --job-name=train_soybean_dino
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --output=slurm-training-%j.out
#SBATCH --account=PAS2699

# TODO 2 Solution: Load modules
module load python/3.10
module load cuda/11.8.0

# TODO 3 Solution: Run the training script
cd $SLURM_SUBMIT_DIR
python train_solution.py
