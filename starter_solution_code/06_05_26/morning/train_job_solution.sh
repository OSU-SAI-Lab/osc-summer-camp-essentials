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
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --output=slurm-training-%j.out
#SBATCH --account=PAS2699

# TODO 2 Solution: Load modules
module load python/3.10
module load cuda/12.1.1
# TODO 3 Solution: Run the training script
cd $SLURM_SUBMIT_DIR

# Maximize PyTorch CPU threading performance
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTORCH_CUDA_ALLOC_CONF="garbage_collection_threshold:0.6,max_split_size_mb:128"

source ../../../venv_osc/bin/activate
python train_solution.py
