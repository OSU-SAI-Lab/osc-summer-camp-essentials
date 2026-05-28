import os

TARGET_DIR = r"c:\Desktop\Coding Projects\osc-summer-camp-essentials\starter_solution_code"

configs = [
    {
        "dir": "06_08_26",
        "name": "error_analysis",
        "time": "00:30:00",
        "desc": "Error Analysis Launch",
        "day": "Monday, June 8 - Morning Session"
    },
    {
        "dir": "06_08_26",
        "name": "hyperparameter_tuning",
        "time": "02:00:00",
        "desc": "Hyperparameter Tuning Launch",
        "day": "Monday, June 8 - Afternoon Session"
    },
    {
        "dir": "06_10_26",
        "name": "edge_optimization",
        "time": "00:30:00",
        "desc": "Edge Optimization Compilation",
        "day": "Wednesday, June 10 - Morning Session"
    },
    {
        "dir": "06_11_26",
        "name": "field_evaluation",
        "time": "00:30:00",
        "desc": "Field Evaluation Launch",
        "day": "Thursday, June 11 - Morning Session"
    }
]

STARTER_TEMPLATE = """#!/bin/bash
# ==============================================================================
# {day}
# SLURM Batch Job Script for {desc}
# ==============================================================================
#
# TODO 1: Set SLURM resources needed for a full GPU training job
# Specifications:
#   - Job name: {name}
#   - Time limit: {time}
#   - Nodes: 1
#   - Cores (ntasks-per-node): 1
#   - CPUs per task (cpus-per-task): 16 (for DataLoader workers)
#   - GPU count: 1 (gpus-per-node=1)
#   - Output file pattern: slurm-{name}-%j.out
#   - Account: PAS2699
#
#SBATCH --job-name={name}
#SBATCH --time={time}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --output=slurm-{name}-%j.out
#SBATCH --account=PAS2699

# ==============================================================================
# TODO 2: Load software modules
# Hint: Load 'python/3.10' and 'cuda/12.1.1'
# Hint: You will also need to activate your virtual environment
# ==============================================================================
# Write module load commands here:



# ==============================================================================
# TODO 3: Execute the PyTorch script
# Hint: You should first `cd $SLURM_SUBMIT_DIR` and then `source ../../venv_osc/bin/activate` before running your script.
# ==============================================================================
# Write your command here:

cd $SLURM_SUBMIT_DIR

# Maximize PyTorch CPU threading performance
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTORCH_CUDA_ALLOC_CONF="garbage_collection_threshold:0.6,max_split_size_mb:128"


"""

SOLUTION_TEMPLATE = """#!/bin/bash
# ==============================================================================
# {day}
# SLURM Batch Job Script for {desc}
# SOLUTION CODE — instructor reference only, do not share with students.
# ==============================================================================
#SBATCH --job-name={name}
#SBATCH --time={time}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --output=slurm-{name}-%j.out
#SBATCH --account=PAS2699

# TODO 2 Solution: Load modules
module load python/3.10
module load cuda/12.1.1

# TODO 3 Solution: Run the script
cd $SLURM_SUBMIT_DIR

# Maximize PyTorch CPU threading performance
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTORCH_CUDA_ALLOC_CONF="garbage_collection_threshold:0.6,max_split_size_mb:128"

source ../../venv_osc/bin/activate
python {name}_solution.py
"""

for c in configs:
    base_dir = os.path.join(TARGET_DIR, c["dir"])
    os.makedirs(base_dir, exist_ok=True)
    
    starter_path = os.path.join(base_dir, f"{c['name']}_job_starter.sh")
    with open(starter_path, "w", newline="\n") as f:
        f.write(STARTER_TEMPLATE.format(**c))
        
    solution_path = os.path.join(base_dir, f"{c['name']}_job_solution.sh")
    with open(solution_path, "w", newline="\n") as f:
        f.write(SOLUTION_TEMPLATE.format(**c))
        
print("All SLURM wrapper scripts generated.")
