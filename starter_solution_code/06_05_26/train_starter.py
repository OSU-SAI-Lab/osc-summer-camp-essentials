"""
Friday, June 5 - Morning Session
Model Training Loop (Starter Code)

What this script is:
A multi-stage model training loop script.

Goal of this script:
Load configuration settings from a YAML file, split the dataset into train/validation sets, train the head first, unfreeze the last block, fine-tune, and save model weight checkpoints.

Why we are doing it (Student Context):
Training is the process of adjusting the model weights so it learns crop patterns. We train the head first to avoid disrupting pre-trained features, then fine-tune the final layers to adapt them to specific crop symptoms.
"""

import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import timm

class SoybeanClassifier(nn.Module):
    def __init__(self, num_classes=6):
        super(SoybeanClassifier, self).__init__()
        print("Initializing SoybeanClassifier with DINOv2 backbone...")
        
        # TODO 2: Initialize backbone with pretrained=True and num_classes=0.
        # Use a try/except block to fallback to 'resnet18' if 'vit_base_patch14_dinov2.lvd142m' fails.
        # Loop through all backbone parameters and freeze them by setting requires_grad = False.
        
        try:
            self.backbone = None  # TODO: Load DINOv2 backbone using timm.create_model
        except Exception as e:
            print(f"Warning: Could not load DINOv2 backbone ({e}). Falling back to ResNet-18.")
            self.backbone = None  # TODO: Load ResNet-18 backbone fallback
            
        in_features = self.backbone.num_features if self.backbone is not None else 512
        self.head = nn.Linear(in_features, num_classes)
        
        # Freeze backbone parameters
        if self.backbone is not None:
            # TODO: Write parameter freezing loop here:
            pass
            
    def forward(self, x):
        # TODO: Implement the forward pass. Pass x through the backbone to extract
        # features, then pass features through the classifier head.
        return None

def main():
    # TODO 1: Load configuration settings from the YAML file relative to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "config_starter.yaml")
    if not os.path.exists(config_path):
        config_path = os.path.join(script_dir, "config_solution.yaml")
        
    # Read config and extract variables:
    # dataset_path, train_ratio, device, batch_size, image_size, epochs_head, lr_head, epochs_fine, lr_fine, weight_decay, save_model_path
    
    # Defaults
    dataset_path = "./temp_soybean_dataset"
    train_ratio = 0.8
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 16
    image_size = 518
    epochs_head = 5
    lr_head = 0.001
    epochs_fine = 3
    lr_fine = 0.00001
    weight_decay = 0.01
    save_model_path = "models/soybean_dinov2_head_model.pth"
    
    # TODO: Load configuration yaml file here and update the variables above:
    
    
    print("=========================================")
    print("Starting DINOv2 Soybean Classifier Training")
    print("=========================================")
    print(f"Device: {device}")
    print(f"Dataset Path: {dataset_path}")
    print(f"Batch Size: {batch_size}")
    
    import sys
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset path '{dataset_path}' not found.")
        print("Are you running this on the OSC cluster? Please check your directory paths in config.yaml.")
        sys.exit(1)
                
    # Define transforms
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    
    # Load dataset twice to apply different transforms to train/val sets in memory
    train_dataset = datasets.ImageFolder(root=dataset_path, transform=train_transform)
    val_dataset = datasets.ImageFolder(root=dataset_path, transform=val_transform)
    
    num_classes = len(train_dataset.classes)
    
    # Generate split indices
    indices = list(range(len(train_dataset)))
    split = int(np.floor(train_ratio * len(train_dataset)))
    np.random.seed(42)  # For reproducible splits
    np.random.shuffle(indices)
    train_idx, val_idx = indices[:split], indices[split:]
    
    train_subset = Subset(train_dataset, train_idx)
    val_subset = Subset(val_dataset, val_idx)
    
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    
    print(f"Loaded {len(train_dataset)} images across {num_classes} classes.")
    print(f"Classes: {train_dataset.classes}")
    print(f"Training split: {len(train_subset)} images, Validation split: {len(val_subset)} images.\n")
    
    # Initialize model
    model = SoybeanClassifier(num_classes=num_classes).to(device)
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # --- PHASE 1: Train Classifier Head ---
    # TODO 3: Implement the training loop for the classifier head.
    # Initialize an Adam optimizer on model.head.parameters() with lr=lr_head.
    # For each epoch: set model to train mode, iterate through train_loader batches,
    # perform zero_grad, forward pass, calculate CrossEntropy loss, backward pass, optimizer step.
    # Note: Print epoch logs in the format: "Epoch X/Y, Loss: Z" for parser compatibility!
    
    print("Training classifier head...")
    optimizer = optim.Adam(model.head.parameters(), lr=lr_head)
    
    for epoch in range(epochs_head):
        model.train()
        running_loss = 0.0
        # TODO: Complete train loader iteration here:
        
        
        # epoch_loss = running_loss / len(train_subset)
        # print(f"Epoch {epoch+1}/{epochs_head}, Loss: {epoch_loss:.4f}")
        
    # --- PHASE 2: Fine-Tuning Backbone ---
    # TODO 4: Unfreeze the last block of the backbone model.
    # Hint: Unfreeze blocks[-1] for ViT backbone or layer4 for ResNet fallback.
    # Then define an AdamW optimizer for all parameters that require gradients with lr=lr_fine, weight_decay=weight_decay.
    # Run the fine-tuning training loop for epochs_fine epochs.
    # Note: Print epoch logs in the format: "Fine Epoch X/Y, Loss: Z" for parser compatibility!
    
    print("\nFine-tuning last transformer block...")
    
    # TODO: Write unfreeze code here:
    
    
    # TODO: Write AdamW optimizer and fine-tuning loop here:
    
        
    print("Fine-tuning complete!")
    
    # TODO 5: Save model checkpoint dictionary including model state_dict and classes list
    # Save to save_model_path.
    
    os.makedirs(os.path.dirname(save_model_path), exist_ok=True)
    # TODO: Save the checkpoint
    # Hint: torch.save({'model_state': model.state_dict(), 'classes': train_dataset.classes}, save_model_path)
    
if __name__ == "__main__":
    main()
