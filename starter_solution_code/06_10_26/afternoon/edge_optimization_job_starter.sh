#!/bin/bash
# ==============================================================================
# Wednesday, June 10 - Morning Session
# SLURM Batch Job Script for Edge Optimization Compilation
# ==============================================================================
#
# TODO 1: Set SLURM resources needed for a full GPU training job
# Specifications:
#   - Job name: edge_optimization
#   - Time limit: 00:30:00
#   - Nodes: 1
#   - Cores (ntasks-per-node): 1
#   - CPUs per task (cpus-per-task): 16 (for DataLoader workers)
#   - GPU count: 1 (gpus-per-node=1)
#   - Output file pattern: slurm-edge_optimization-%j.out
#   - Account: PAS2699
#
#SBATCH --job-name=edge_optimization
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --output=slurm-edge_optimization-%j.out
#SBATCH --account=PAS2699

# ==============================================================================
# TODO 2: Load software modules
# Hint: Load 'python/3.10' and 'cuda/12.1.1'
# Hint: You will also need to activate your virtual environment
# ==============================================================================
# Write module load commands here:



# ==============================================================================
# TODO 3: Execute the PyTorch script
# Hint: You should first `cd $SLURM_SUBMIT_DIR` and then `source ../../../venv_osc/bin/activate` before running your script.
# ==============================================================================
# Write your command here:

cd $SLURM_SUBMIT_DIR

# Maximize PyTorch CPU threading performance
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTORCH_CUDA_ALLOC_CONF="garbage_collection_threshold:0.6,max_split_size_mb:128"


