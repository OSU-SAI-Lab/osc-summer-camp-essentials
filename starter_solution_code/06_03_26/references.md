# Wednesday, June 3: OSC Foundations & Exploratory Data Analysis

Welcome to Day 1 of the AI in the Field Camp! Today we will get comfortable with the Ohio Supercomputer Center (OSC) environment, check our GPU availability, and construct PyTorch DataLoaders to explore our Soybean disease dataset.

---

## Morning Session: OSC Foundations & "Hello World" GPU Node

### Useful Commands

To run scripts on OSC Ascend or Cardinal, we load the required pre-installed software modules. In your terminal, run:

```bash
# Load the Python and PyTorch modules
module load python/3.10
module load cuda/12.1.1

# Create a virtual environment
python -m venv venv_osc

# Install requirements
pip install -r requirements.txt

# Activate the virtual environment
source ../../../venv_osc/bin/activate

# Verify the loaded modules
module list
```

### Submitting a Job to the GPU Node (SLURM)

Because training deep learning models requires massive compute power, we run our code on dedicated GPU nodes. We submit jobs to these nodes using SLURM batch scripts.

* `sbatch hello_gpu_job.sh`: Submit your job to the cluster.
* `squeue -u <your_username>`: Check the status of your submitted job (e.g., `PD` = Pending, `R` = Running).
* `scancel <job_id>`: Cancel a running or pending job.
* `cat slurm-<job_id>.out`: View the standard output log of your job.

---

## Afternoon Session: Exploratory Data Analysis (EDA) & DataLoaders

### How PyTorch Sees Datasets

PyTorch uses the `torchvision.datasets.ImageFolder` class to load datasets structured as folders containing images. For example, our dataset has this directory structure:

```text
Soybean_Dataset/
├── DicambaDamage/
│   ├── image_001.jpg
│   └── ...
├── FrogEyeLeafSpot/
│   └── ...
├── GenericFeeding/
│   └── ...
├── InsectDamage/
│   └── ...
├── Soybeans/
│   └── ...
└── SuddenDeathSyndrome/
    └── ...
```

When we load this using `ImageFolder`, PyTorch automatically maps each folder name to an integer label (e.g., `0` for `DicambaDamage`, `1` for `FrogEyeLeafSpot`, etc.).

### PyTorch Image Transforms

Transforms allow us to resize, normalize, and augment our images. Common transforms include:

* `transforms.Resize((width, height))`: Standardizes all crop images to the same resolution. DINOv2 requires image inputs to be resized to multiples of 14, typically `(518, 518)`.
* `transforms.RandomHorizontalFlip()`: Randomly flips the image horizontally to teach the model that disease indicators can appear in any orientation.
* `transforms.RandomRotation(degrees)`: Rotates the image randomly to increase model robustness against camera angles.
* `transforms.ToTensor()`: Converts PIL Images into PyTorch tensors and normalizes pixel values to the `[0, 1]` range.

---

## Curated Resources for Day 1

* [OSC Slurm Guide](https://bpb-us-w2.wpmucdn.com/u.osu.edu/dist/4/102442/files/2025/03/OSC-Slurm-Guideline.pdf)
* [Official PyTorch Transforms Documentation](https://pytorch.org/vision/stable/transforms.html)
* [Intro to PyTorch DataLoaders](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html)
* [Matplotlib Pyplot Tutorial](https://matplotlib.org/stable/tutorials/introductory/pyplot.html)
