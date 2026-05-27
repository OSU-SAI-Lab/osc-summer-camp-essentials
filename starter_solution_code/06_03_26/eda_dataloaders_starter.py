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
import shutil
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

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
    # Set the dataset root path.
    # On OSC: use /fs/ess/PAS2699/AI_Presidency_Dataset_CSG/Soybeans/Soybeans
    # On your local computer: fallback to a temporary mock directory
    OSC_PATH = "/fs/ess/PAS2699/AI_Presidency_Dataset_CSG/Soybeans/Soybeans"
    DATASET_ROOT = OSC_PATH if os.path.exists(OSC_PATH) else "./temp_soybean_dataset"
    
    if DATASET_ROOT == "./temp_soybean_dataset" and not os.path.exists(DATASET_ROOT):
        create_mock_dataset(DATASET_ROOT)

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

    # Clean up mock dataset if created
    if DATASET_ROOT == "./temp_soybean_dataset" and os.path.exists(DATASET_ROOT):
        # Optionally remove the directory afterwards or leave it for student inspection
        print("\nNote: Temporary local dataset folder is saved at './temp_soybean_dataset'.")

if __name__ == "__main__":
    main()
