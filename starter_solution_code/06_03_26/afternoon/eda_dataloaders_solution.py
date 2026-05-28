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
import sys
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from collections import Counter

def main():
    DATASET_ROOT = "/fs/ess/PAS2699/AI_Presidency_Dataset_CSG/Soybeans/Soybeans"
    
    if not os.path.exists(DATASET_ROOT):
        print(f"Error: Dataset path '{DATASET_ROOT}' not found.")
        print("Are you running this on the OSC cluster? Please check your directory paths.")
        sys.exit(1)

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

if __name__ == "__main__":
    main()
