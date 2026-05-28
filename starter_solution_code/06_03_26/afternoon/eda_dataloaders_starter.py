"""
Wednesday, June 3 - Afternoon Session
Exploratory Data Analysis (EDA) and DataLoaders (Starter Code)

What this script is:
An Exploratory Data Analysis (EDA) and PyTorch dataset loader script.

Goal of this script:
Compose image transforms (resize, flip, rotate), load images using ImageFolder, calculate class balances to detect imbalance, and visualize a batch of augmented data.

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

    # TODO 1: Define your training and testing image transforms
    # requirements:
    #   - train_transform: Resize to (518, 518), Random Horizontal Flip,
    #                      Random Rotation (up to 15 degrees), convert to Tensor.
    #   - test_transform:  Resize to (518, 518), convert to Tensor (no augmentations).
    
    train_transform = transforms.Compose([
        # TODO: Add transforms here
    
    ])

    test_transform = transforms.Compose([
        # TODO: Add transforms here
    
    ])

    # TODO 2: Load the dataset using torchvision.datasets.ImageFolder
    # Hint: Use the train_transform for training data
    
    print("Loading dataset using ImageFolder...")
    dataset = None # TODO: Load dataset using ImageFolder
    print(f"Successfully loaded dataset! Classes found: {dataset.classes}")

    # TODO 3: Calculate and print the class distribution (imbalance check)
    # Count how many images are in each class folder.
    # Hint: Look through the dataset.targets (which are class indices for each image)
    
    print("\nCalculating class distributions...")
    class_counts = {}
    # TODO: Fill in the class_counts dictionary mapping class names to image counts
    
    for cls_name, count in class_counts.items():
        print(f"  - {cls_name}: {count} images")

    # TODO 4: Initialize a PyTorch DataLoader
    # Hint: Use DataLoader(dataset, batch_size=4, shuffle=True)
    
    dataloader = None # TODO: Initialize DataLoader
    print("\nDataLoader initialized.")

    # TODO 5: Fetch one batch of data and plot the images using matplotlib
    # To plot a PyTorch tensor, you need to:
    #   1. Permute the dimensions from (Channels, Height, Width) to (Height, Width, Channels)
    #   2. Convert to numpy array
    
    print("Fetching sample batch for visualization...")
    images, labels = next(iter(dataloader))

    plt.figure(figsize=(12, 6))
    for i in range(4):
        # TODO: Get the image tensor, permute it to (H, W, C), and show it
        # Hint: use img_tensor.permute(1, 2, 0).numpy()
        
        img = None # TODO: Permute tensor to numpy
        
        plt.subplot(1, 4, i + 1)
        plt.imshow(img)
        plt.title(f"Class: {dataset.classes[labels[i].item()]}")
        plt.axis("off")
        
    plt.tight_layout()
    plt.suptitle("Augmented Training Image Samples from DataLoader", fontsize=16)
    plt.show()

if __name__ == "__main__":
    main()
