"""
Thursday, June 11 - Afternoon Session
Generalization and Domain-Shift Evaluation (Starter Code)

What this script is:
An out-of-distribution (OOD) accuracy evaluation script.

Goal of this script:
Compare model accuracy on validation (in-distribution) data vs. newly captured mobile phone (out-of-distribution) data, and plot comparative bar charts.

Why we are doing it (Student Context):
Models often fail in the real world due to domain shift (shadows, angles, camera quality). Evaluating this out-of-distribution accuracy helps us understand if the model generalizes.
"""

import os
import sys
import yaml
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import timm
from tqdm import tqdm

# ==============================================================================
# Helper Code Provided by Instructor
# We have provided the PyTorch evaluation loop for you to evaluate the real model.
# ==============================================================================
class SoybeanClassifier(nn.Module):
    def __init__(self, num_classes=6):
        super(SoybeanClassifier, self).__init__()
        try:
            self.backbone = timm.create_model('vit_base_patch14_dinov2.lvd142m', pretrained=False, num_classes=0)
        except Exception:
            self.backbone = timm.create_model('resnet18', pretrained=False, num_classes=0)
        in_features = self.backbone.num_features
        self.head = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        features = self.backbone(x)
        logits = self.head(features)
        return logits

def get_loaders():
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
    
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    
    # 1. Load Validation Dataset
    if not os.path.exists(dataset_path):
        print(f"Error: Training dataset path '{dataset_path}' not found.")
        sys.exit(1)
    
    full_dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
    indices = list(range(len(full_dataset)))
    split = int(np.floor(train_ratio * len(full_dataset)))
    np.random.seed(42)
    np.random.shuffle(indices)
    _, val_idx = indices[:split], indices[split:]
    val_subset = Subset(full_dataset, val_idx)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    
    # 2. Load Field Dataset
    FIELD_DATA_DIR = "/fs/ess/PAS2699/AI_Presidency_Dataset_CSG/Soybeans/Field_Data"
    if not os.path.exists(FIELD_DATA_DIR):
        print(f"Error: Field dataset path '{FIELD_DATA_DIR}' not found.")
        print("Please update FIELD_DATA_DIR in the script to point to the images collected on Day 2.")
        sys.exit(1)
        
    field_dataset = datasets.ImageFolder(root=FIELD_DATA_DIR, transform=transform)
    field_loader = DataLoader(field_dataset, batch_size=batch_size, shuffle=False)
    
    return val_loader, field_loader, len(full_dataset.classes)

def evaluate_loader(model, loader, device, desc="Evaluating"):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(loader, desc=desc):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

def run_field_evaluation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading datasets on {device}...")
    val_loader, field_loader, num_classes = get_loaders()
    
    model = SoybeanClassifier(num_classes=num_classes)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "../../06_05_26/morning/models/soybean_dinov2_head_model.pth")
    
    if os.path.exists(model_path):
        print(f"Loading weights from {model_path}...")
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and "model_state" in checkpoint:
            model.load_state_dict(checkpoint["model_state"])
        else:
            model.load_state_dict(checkpoint)
    else:
        print(f"Warning: {model_path} not found. Running with random weights.")
        
    model = model.to(device)
    
    val_accuracy = evaluate_loader(model, val_loader, device, "Evaluating Validation Set (ID)")
    field_accuracy = evaluate_loader(model, field_loader, device, "Evaluating Field Set (OOD)")
    return val_accuracy, field_accuracy
# ==============================================================================


def main():
    field_accuracy = 0.0 # TODO: Calculate field accuracy

    print("=========================================")
    print("Out-of-Distribution (OOD) Evaluation:")
    print("=========================================")
    print(f"Validation (In-Distribution) Accuracy: {val_accuracy * 100:.2f}%")
    print(f"Field Test (Out-of-Distribution) Accuracy: {field_accuracy * 100:.2f}%")
    print(f"Accuracy Gap (Domain Shift): {(val_accuracy - field_accuracy) * 100:.2f}%\n")

    # TODO 3: Plot a comparative bar chart of the two accuracies
    # Requirements:
    #   - Customize bar colors
    #   - Add gridlines, title, axis labels
    #   - Set y-limit [0, 1.0]
    #   - Annotate values on top of the bars
    
    labels = ['Validation Set (ID)', 'Field Test Set (OOD)']
    accuracies = [val_accuracy, field_accuracy]

    plt.figure(figsize=(8, 5))
    
    # TODO: Write matplotlib comparative bar chart code here
    
    
    # Save the evaluation plot
    os.makedirs("outputs", exist_ok=True)
    output_img = "outputs/field_vs_val_comparison.png"
    plt.savefig(output_img, dpi=300)
    print(f"Successfully generated and saved plot to '{output_img}'.")
    plt.show()

if __name__ == "__main__":
    main()
