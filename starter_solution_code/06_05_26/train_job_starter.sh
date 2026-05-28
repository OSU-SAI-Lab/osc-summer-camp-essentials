#!/bin/bash
# ==============================================================================
# Friday, June 5 - Morning Session
# SLURM Batch Job Script for DINOv2 Training Launch
# ==============================================================================
#
# TODO 1: Set SLURM resources needed for a full GPU training job
# Specifications:
#   - Job name: train_soybean_dino
#   - Time limit: 2 hours (02:00:00)
#   - Nodes: 1
#   - Cores (ntasks-per-node): 1
#   - CPUs per task (cpus-per-task): 8 (for DataLoader workers)
#   - GPU count: 1 (gpus-per-node=1)
#   - Output file pattern: slurm-training-%j.out
#   - Account: PAS2699
#
#SBATCH --job-name=train_soybean_dino
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --output=slurm-training-%j.out
#SBATCH --account=PAS2699

# ==============================================================================
# TODO 2: Load software modules
# Hint: Load 'python/3.10' and 'cuda/12.1.1'
# Hint: You will also need to activate your virtual environment (e.g., source ../../venv_osc/bin/activate)
# ==============================================================================
# Write module load commands here:
module load python/3.10
module load cuda/12.1.1



# ==============================================================================
# TODO 3: Execute the DINOv2 training script
# Hint: You should first `cd $SLURM_SUBMIT_DIR` and then `source ../../venv_osc/bin/activate` before running your script.
# ==============================================================================
# Write your command here:

cd $SLURM_SUBMIT_DIR

# Maximize PyTorch CPU threading performance
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTORCH_CUDA_ALLOC_CONF="garbage_collection_threshold:0.6,max_split_size_mb:128"

source ../../venv_osc/bin/activate
python train_starter.py
