#!/bin/bash
# ==============================================================================
# Wednesday, June 3 - Morning Session
# SLURM Batch Job Script for GPU Node Verification
# ==============================================================================
#
# TODO 1: Set SLURM resources needed for a GPU job
# Hint: You need to specify:
#   - Job name
#   - Time limit (e.g. 10 minutes: 00:10:00)
#   - Node count (1 node is enough)
#   - GPU count (1 GPU is enough: --gpus-per-node=1)
#   - Output file pattern (e.g. slurm-%j.out)
#   - OSC Project Account (use PAS2699 as default or your class code)
#
#SBATCH --job-name=hello_gpu_verification
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --output=slurm-%j.out
#SBATCH --account=PAS2699

# ==============================================================================
# TODO 2: Load the required software modules
# Hint: Load 'python/3.10' and 'cuda/12.1.1'
# Hint: You will also need to activate your virtual environment (e.g., source ../../venv_osc/bin/activate)
# ==============================================================================
# Write module load commands here:



# ==============================================================================
# TODO 3: Execute your python verification script
# Hint: You should first `cd $SLURM_SUBMIT_DIR` and then `source ../../venv_osc/bin/activate` before running your script.
# ==============================================================================
# Write the execution command here:

