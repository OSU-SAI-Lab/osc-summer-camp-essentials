"""
Wednesday, June 3 - Afternoon Session
Exploratory Data Analysis (EDA) and DataLoaders (Solution Code)

What this script is:
A reference solution for Exploratory Data Analysis (EDA) and dataset loading.

Goal of this script:
Implement image transforms, load data using ImageFolder, compute class distributions, and plot a sample batch.

Why we are doing it (Student Context):
AI models learn from data. Inspecting data for class balance and setting up data loaders to pipe augmented crop images is the first step to prevent model bias and ensure it generalizes to different crop scenes.
"""

import os
import shutil
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from collections import Counter

def create_mock_dataset(root_dir):
    """
    Helper function to create a small mock crop disease dataset.
    This ensures the code runs locally on your computer even if you are not at OSC.
    """
    classes = ['DicambaDamage', 'FrogEyeLeafSpot', 'GenericFeeding', 'InsectDamage', 'Soybeans', 'SuddenDeathSyndrome']
    os.makedirs(root_dir, exist_ok=True)
    
    # Generate 5 sample images per class
    for cls in classes:
        cls_dir = os.path.join(root_dir, cls)
        os.makedirs(cls_dir, exist_ok=True)
        for i in range(5):
            # Create a simple color block image representing a leaf
            img_arr = np.random.randint(50, 200, size=(100, 100, 3), dtype=np.uint8)
            # Add a colored green shape to make it look like a leaf
            img_arr[30:70, 30:70, 1] = 220 # Add green channel boost
            img = Image.fromarray(img_arr)
            img.save(os.path.join(cls_dir, f"leaf_{i}.jpg"))
            
    print(f"Created temporary mock dataset with {len(classes)} classes in '{root_dir}'.\n")

def main():
    OSC_PATH = "/fs/ess/PAS2699/AI_Presidency_Dataset_CSG/Soybeans/Soybeans"
    DATASET_ROOT = OSC_PATH if os.path.exists(OSC_PATH) else "./temp_soybean_dataset"
    
    if DATASET_ROOT == "./temp_soybean_dataset" and not os.path.exists(DATASET_ROOT):
        create_mock_dataset(DATASET_ROOT)

    # TODO 1 Solution: Define transforms
    train_transform = transforms.Compose([
        transforms.Resize((518, 518)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((518, 518)),
        transforms.ToTensor(),
    ])

    # TODO 2 Solution: Load dataset using ImageFolder
    print("Loading dataset using ImageFolder...")
    dataset = datasets.ImageFolder(root=DATASET_ROOT, transform=train_transform)
    print(f"Successfully loaded dataset! Classes found: {dataset.classes}")

    # TODO 3 Solution: Calculate class distributions
    print("\nCalculating class distributions...")
    targets = dataset.targets
    counts = Counter(targets)
    class_counts = {dataset.classes[idx]: count for idx, count in counts.items()}
    
    for cls_name, count in class_counts.items():
        print(f"  - {cls_name}: {count} images")

    # TODO 4 Solution: Initialize DataLoader
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    print("\nDataLoader initialized.")

    # TODO 5 Solution: Fetch one batch and plot
    print("Fetching sample batch for visualization...")
    images, labels = next(iter(dataloader))

    plt.figure(figsize=(12, 6))
    for i in range(4):
        # Permute from (C, H, W) to (H, W, C)
        img = images[i].permute(1, 2, 0).numpy()
        
        plt.subplot(1, 4, i + 1)
        plt.imshow(img)
        plt.title(f"Class: {dataset.classes[labels[i].item()]}")
        plt.axis("off")
        
    plt.tight_layout()
    plt.suptitle("Augmented Training Image Samples from DataLoader", fontsize=16)
    plt.show()

    if DATASET_ROOT == "./temp_soybean_dataset" and os.path.exists(DATASET_ROOT):
        print("\nNote: Temporary local dataset folder is saved at './temp_soybean_dataset'.")

if __name__ == "__main__":
    main()
