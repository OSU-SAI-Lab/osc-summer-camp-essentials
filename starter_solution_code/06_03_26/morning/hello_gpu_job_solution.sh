#!/bin/bash
# ==============================================================================
# Wednesday, June 3 - Morning Session
# SLURM Batch Job Script for GPU Node Verification
# SOLUTION CODE — instructor reference only, do not share with students.
# ==============================================================================
#SBATCH --job-name=hello_gpu_verification
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --output=slurm-%j.out
#SBATCH --account=PAS2699

# TODO 2 Solution: Load modules
module load python/3.10
module load cuda/12.1.1
# TODO 3 Solution: Run python script
cd $SLURM_SUBMIT_DIR
source ../../../venv_osc/bin/activate
python hello_gpu_solution.py
