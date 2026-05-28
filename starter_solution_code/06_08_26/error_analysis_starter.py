"""
Monday, June 8 - Morning Session
Error Analysis and Metrics Report (Starter Code)

What this script is:
A metrics report generator and confusion matrix plotter.

Goal of this script:
Generate classification reports (Precision, Recall, F1-Scores) for each disease class, and plot a confusion matrix heatmap.

Why we are doing it (Student Context):
A simple accuracy score hides model mistakes. Precision and recall tell us exactly which crop classes (e.g. Insect Damage vs Healthy) the model confuses, guiding our troubleshooting and data improvements.
"""

import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import timm

# ==============================================================================
# Helper Code Provided by Instructor
# We have already written the PyTorch inference loop for you so you can focus 
# on the error analysis metrics today. You will learn how to write PyTorch 
# inference loops yourself tomorrow!
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

def get_val_loader():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "../06_05_26/config_starter.yaml")
    if not os.path.exists(config_path):
        config_path = os.path.join(script_dir, "../06_05_26/config_solution.yaml")
        
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    dataset_path = config['dataset']['root_path']
    train_ratio = config['dataset']['train_ratio']
    batch_size = config['environment'].get('batch_size') or 32
    image_size = config['environment'].get('image_size') or 518
    
    if dataset_path == "./temp_soybean_dataset" or not os.path.exists(dataset_path):
        print(f"Warning: Dataset path '{dataset_path}' not found. Falling back to local temp dataset.")
        dataset_path = os.path.join(script_dir, "../06_05_26/temp_soybean_dataset")
        
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    
    val_dataset = datasets.ImageFolder(root=dataset_path, transform=val_transform)
    classes = val_dataset.classes
    
    indices = list(range(len(val_dataset)))
    split = int(np.floor(train_ratio * len(val_dataset)))
    np.random.seed(42)  # Critical: Use exact same seed as training script!
    np.random.shuffle(indices)
    _, val_idx = indices[:split], indices[split:]
    
    val_subset = Subset(val_dataset, val_idx)
    val_loader = DataLoader(val_subset, batch_size=min(16, batch_size), shuffle=False)
    
    return val_loader, classes

def run_inference_helper():
    print("Loading Validation Dataset...")
    val_loader, classes = get_val_loader()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = SoybeanClassifier(num_classes=len(classes))
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "../06_05_26/models/soybean_dinov2_head_model.pth")
    
    if os.path.exists(model_path):
        print(f"Loading weights from {model_path}...")
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and "model_state" in checkpoint:
            model.load_state_dict(checkpoint["model_state"])
        else:
            model.load_state_dict(checkpoint)
    else:
        print(f"Warning: {model_path} not found. Running with random weights for demo.")
        
    model = model.to(device)
    model.eval()
    
    y_true = []
    y_pred = []
    
    print("Running validation inference. This may take a moment...")
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            y_true.extend(labels.numpy())
            y_pred.extend(preds.cpu().numpy())
            
    return np.array(y_true), np.array(y_pred), classes
# ==============================================================================


def main():
    # Load actual true labels and predicted labels by running the model
    y_true, y_pred, classes = run_inference_helper()

    # TODO 1: Generate and print the text-based classification report
    # Hint: Use classification_report(y_true, y_pred, target_names=classes)
    
    print("=========================================")
    print("Classification Metrics Report:")
    print("=========================================")
    report = "" # TODO: Generate report
    print(report)

    # TODO 2: Calculate the confusion matrix
    # Hint: Use confusion_matrix(y_true, y_pred)
    
    cm = None # TODO: Calculate confusion matrix
    print("Confusion Matrix calculated.")

    # TODO 3: Plot the confusion matrix as an annotated heatmap using Matplotlib
    # Requirements:
    #   - Use plt.imshow() to show the matrix values
    #   - Add class labels to x-axis and y-axis ticks
    #   - Loop through the matrix and add text annotations inside each cell
    #   - Add axis labels and title
    
    plt.figure(figsize=(10, 8))
    
    # TODO: Write matplotlib heatmap plotting code here
    
    
    # Save the heatmap plot
    os.makedirs("outputs", exist_ok=True)
    output_img = "outputs/confusion_matrix.png"
    plt.savefig(output_img, dpi=300)
    print(f"\nSuccessfully generated and saved plot to '{output_img}'.")
    plt.show()

if __name__ == "__main__":
    main()
