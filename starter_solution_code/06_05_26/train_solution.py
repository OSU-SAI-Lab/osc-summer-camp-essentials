"""
Friday, June 5 - Morning Session
Model Training Loop (Solution Code)

What this script is:
A reference solution for the multi-stage training loop.

Goal of this script:
Load YAML configurations, split the dataset, train the classification head, unfreeze the last block, fine-tune the backbone, and save the resulting checkpoint.

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
        try:
            self.backbone = timm.create_model('vit_base_patch14_dinov2.lvd142m', pretrained=True, num_classes=0)
        except Exception as e:
            print(f"Warning: Could not load DINOv2 backbone ({e}). Falling back to ResNet-18.")
            self.backbone = timm.create_model('resnet18', pretrained=True, num_classes=0)
            
        in_features = self.backbone.num_features
        self.head = nn.Linear(in_features, num_classes)
        
        # Freeze backbone parameters
        for param in self.backbone.parameters():
            param.requires_grad = False
            
    def forward(self, x):
        features = self.backbone(x)
        logits = self.head(features)
        return logits

def main():
    # Load configuration relative to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "config_solution.yaml")
    if not os.path.exists(config_path):
        config_path = os.path.join(script_dir, "config_starter.yaml")
        
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    dataset_path = config['dataset']['root_path']
    train_ratio = config['dataset']['train_ratio']
    device = torch.device(config['environment']['device'] if torch.cuda.is_available() else "cpu")
    batch_size = config['environment']['batch_size']
    image_size = config['environment']['image_size']
    
    epochs_head = config['hyperparameters']['epochs_head']
    lr_head = config['hyperparameters']['lr_head']
    epochs_fine = config['hyperparameters']['epochs_fine']
    lr_fine = config['hyperparameters']['lr_fine']
    weight_decay = config['hyperparameters']['weight_decay']
    save_model_path = config['output']['save_model_path']
    
    print("=========================================")
    print("Starting DINOv2 Soybean Classifier Training")
    print("=========================================")
    print(f"Device: {device}")
    print(f"Dataset Path: {dataset_path}")
    print(f"Batch Size: {batch_size}")
    
    # Fallback if dataset path does not exist (e.g. running locally instead of OSC)
    if not os.path.exists(dataset_path):
        print(f"Warning: Dataset path '{dataset_path}' not found. Falling back to local './temp_soybean_dataset'.")
        dataset_path = "./temp_soybean_dataset"
    
    # Create temp mock dataset if local and doesn't exist
    if dataset_path == "./temp_soybean_dataset" and not os.path.exists(dataset_path):
        classes = ['DicambaDamage', 'FrogEyeLeafSpot', 'GenericFeeding', 'InsectDamage', 'Soybeans', 'SuddenDeathSyndrome']
        os.makedirs(dataset_path, exist_ok=True)
        for cls in classes:
            cls_dir = os.path.join(dataset_path, cls)
            os.makedirs(cls_dir, exist_ok=True)
            for i in range(15):
                img_arr = np.random.randint(50, 200, size=(100, 100, 3), dtype=np.uint8)
                img_arr[30:70, 30:70, 1] = 220
                from PIL import Image
                Image.fromarray(img_arr).save(os.path.join(cls_dir, f"leaf_{i}.jpg"))
                
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
    
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=8)
    
    print(f"Loaded {len(train_dataset)} images across {num_classes} classes.")
    print(f"Classes: {train_dataset.classes}")
    print(f"Training split: {len(train_subset)} images, Validation split: {len(val_subset)} images.\n")
    
    # Initialize model
    model = SoybeanClassifier(num_classes=num_classes).to(device)
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # --- PHASE 1: Train Classifier Head ---
    print("Training classifier head...")
    optimizer = optim.Adam(model.head.parameters(), lr=lr_head)
    
    for epoch in range(epochs_head):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            
        epoch_loss = running_loss / len(train_subset)
        print(f"Epoch {epoch+1}/{epochs_head}, Loss: {epoch_loss:.4f}")
        
    # --- PHASE 2: Fine-Tuning Backbone ---
    print("\nFine-tuning last transformer block...")
    # Unfreeze the last block of the backbone
    if hasattr(model.backbone, 'blocks') and len(model.backbone.blocks) > 0:
        # ViT blocks
        for param in model.backbone.blocks[-1].parameters():
            param.requires_grad = True
        print("Unfroze last transformer block (blocks[-1]).")
    elif hasattr(model.backbone, 'layer4'):
        # ResNet layer4
        for param in model.backbone.layer4.parameters():
            param.requires_grad = True
        print("Unfroze ResNet block (layer4).")
            
    # Optimizer for fine-tuning (all trainable parameters get lr_fine)
    optimizer_fine = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr_fine,
        weight_decay=weight_decay
    )
    
    for epoch in range(epochs_fine):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer_fine.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer_fine.step()
            running_loss += loss.item() * images.size(0)
            
        epoch_loss = running_loss / len(train_subset)
        print(f"Fine Epoch {epoch+1}/{epochs_fine}, Loss: {epoch_loss:.4f}")
        
    print("Fine-tuning complete!")
    
    # Save the model
    os.makedirs(os.path.dirname(save_model_path), exist_ok=True)
    torch.save({
        'model_state': model.state_dict(),
        'classes': train_dataset.classes
    }, save_model_path)
    print(f"Model saved successfully to '{save_model_path}'")
    
if __name__ == "__main__":
    main()
