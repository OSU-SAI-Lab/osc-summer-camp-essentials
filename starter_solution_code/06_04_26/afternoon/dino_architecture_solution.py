"""
Thursday, June 4 - Afternoon Session
Model Architecture Definition (Solution Code)

What this script is:
A reference solution for defining the transfer learning model.

Goal of this script:
Define the SoybeanClassifier model class using a pre-trained, frozen DINOv2 model and a trainable classification layer.

Why we are doing it (Student Context):
Training deep vision networks from scratch takes days and thousands of images. Transfer learning allows us to reuse pre-trained feature representation models (DINOv2) and only train a classification head, saving compute and time.
"""

import torch
import torch.nn as nn
import timm

class SoybeanClassifier(nn.Module):
    def __init__(self, num_classes=6):
        super(SoybeanClassifier, self).__init__()
        
        print("Initializing SoybeanClassifier...")
        
        # TODO 1 Solution: Load pretrained backbone
        self.backbone = timm.create_model('vit_base_patch14_dinov2.lvd142m', pretrained=True, num_classes=0)
        
        # TODO 2 Solution: Freeze backbone parameters
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Extract features dimension from backbone
        in_features = self.backbone.num_features
        print(f"DINOv2 Backbone backbone features shape: {in_features}")
        
        # TODO 3 Solution: Define linear classifier head
        self.head = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        # TODO 4 Solution: Forward pass logic
        features = self.backbone(x)
        logits = self.head(features)
        return logits

def main():
    # Number of target classes: 6
    NUM_CLASSES = 6
    
    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    try:
        # Instantiate model
        model = SoybeanClassifier(num_classes=NUM_CLASSES)
        model = model.to(device)
        print("Model initialized successfully!")
        
        # Print parameter counts
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total Parameters: {total_params:,}")
        print(f"Trainable Parameters (Classifier Head only): {trainable_params:,}\n")
        
        # Verify shape using a dummy image tensor
        print("Running forward pass with dummy tensor...")
        dummy_input = torch.randn(1, 3, 518, 518).to(device)
        
        model.eval()
        with torch.no_grad():
            outputs = model(dummy_input)
            
        print(f"Dummy Input shape: {dummy_input.shape}")
        print(f"Model Output shape: {outputs.shape}")
        
        # Check output correctness
        assert outputs.shape == (1, NUM_CLASSES), f"Expected output shape (1, {NUM_CLASSES}), but got {outputs.shape}"
        print("SUCCESS! Output shape verification matches expected classes.")
        
    except Exception as e:
        print(f"\nVerification FAILED: {e}")

if __name__ == "__main__":
    main()
