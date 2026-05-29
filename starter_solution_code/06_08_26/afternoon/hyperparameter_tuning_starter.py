"""
Monday, June 8 - Afternoon Session
Hyperparameter Sweep Simulation (Starter Code)

What this script is:
A hyperparameter search and evaluation script.

Goal of this script:
Simulate training with different learning rates, save sweep results to JSON, and plot comparison charts.

Why we are doing it (Student Context):
Hyperparameters (like learning rate) determine how models learn. We conduct systematic sweeps to find the optimal settings rather than guessing, ensuring our model converges reliably.
"""

import json
import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import timm
import matplotlib.pyplot as plt
from tqdm import tqdm

# ==============================================================================
# Helper Code Provided by Instructor
# We have already written the PyTorch sweep loop for you so you can focus 
# on the logging and chart visualization!
# ==============================================================================
class SoybeanClassifier(nn.Module):
    def __init__(self, num_classes=6):
        super(SoybeanClassifier, self).__init__()
        try:
            self.backbone = timm.create_model('vit_base_patch14_dinov2.lvd142m', pretrained=True, num_classes=0)
        except Exception:
            self.backbone = timm.create_model('resnet18', pretrained=True, num_classes=0)
        in_features = self.backbone.num_features
        self.head = nn.Linear(in_features, num_classes)
        for param in self.backbone.parameters():
            param.requires_grad = False
            
    def forward(self, x):
        features = self.backbone(x)
        logits = self.head(features)
        return logits

def get_dataloaders():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "../../06_05_26/morning/config_starter.yaml")
    if not os.path.exists(config_path):
        config_path = os.path.join(script_dir, "../../06_05_26/morning/config_solution.yaml")
        
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    dataset_path = config['dataset']['root_path']
    train_ratio = config['dataset']['train_ratio']
    batch_size = config['environment'].get('batch_size') or 32
    image_size = config['environment'].get('image_size') or 518
    
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset path '{dataset_path}' not found. Please check your OSC path.")
        sys.exit(1)
        
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    
    dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
    
    indices = list(range(len(dataset)))
    split = int(np.floor(train_ratio * len(dataset)))
    np.random.seed(42)
    np.random.shuffle(indices)
    train_idx, val_idx = indices[:split], indices[split:]
    
    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)
    
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, val_loader, len(dataset.classes)

# Load global dataset variables once
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Preparing datasets on {DEVICE}...")
TRAIN_LOADER, VAL_LOADER, NUM_CLASSES = get_dataloaders()

def train_evaluate_lr(lr):
    """
    Trains the head for 1 epoch with a specific learning rate, and returns validation metrics.
    """
    model = SoybeanClassifier(NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.head.parameters(), lr=lr)
    
    model.train()
    print(f"  Training 1 epoch with lr={lr}...")
    for images, labels in tqdm(TRAIN_LOADER, leave=False):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in VAL_LOADER:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    return correct / total, val_loss / total
# ==============================================================================

def main():
    # TODO 1: Define search space
    # Hint: Create a list called `learning_rates` containing [1e-2, 1e-3, 1e-4]
    learning_rates = [1e-2, 1e-3, 1e-4]

    results = []

    print("Starting actual PyTorch hyperparameter sweep (1 epoch each)...")
    for lr in learning_rates:
        print(f"\n--- Running experiment with learning_rate={lr} ---")
        
        # Run PyTorch Evaluation Helper
        acc, loss = train_evaluate_lr(lr)
        
        # Store trial results
        trial_data = {
            "learning_rate": lr,
            "validation_accuracy": acc,
            "validation_loss": loss
        }
        results.append(trial_data)
        print(f"  Result: Accuracy={acc:.4f}, Loss={loss:.4f}")

    # TODO 2: Save to JSON file
    # Hint: Use json.dump(results, f, indent=4) to save the results array to "outputs/tuning_results.json"
    os.makedirs("outputs", exist_ok=True)
    output_json = "outputs/tuning_results.json"
    
    # Write JSON saving logic here:
    with open(output_json, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\nAll trial results saved to '{output_json}'.")

    # Extract values for plotting
    lr_labels = [str(r["learning_rate"]) for r in results]
    accuracies = [r["validation_accuracy"] for r in results]

    # TODO 3: Plot a bar chart comparing validation accuracy across learning rates
    # Requirements:
    #   - Set bar labels as string values of learning rates
    #   - Customize bar colors, add grid, y-axis limit [0, 1]
    #   - Add title and axis labels
    
    plt.figure(figsize=(8, 5))
    
    # TODO: Write matplotlib bar chart plotting code here
    
    
    # Save the chart
    output_img = "outputs/tuning_comparison.png"
    plt.savefig(output_img, dpi=300)
    print(f"Successfully generated and saved plot to '{output_img}'.")
    plt.show()

if __name__ == "__main__":
    main()
