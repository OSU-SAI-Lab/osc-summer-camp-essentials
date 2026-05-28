"""
Thursday, June 4 - Afternoon Session
Model Architecture Definition (Starter Code)

What this script is:
A transfer learning model architecture definition script.

Goal of this script:
Load a pre-trained DINOv2 backbone as a feature extractor, freeze its weights, append a classification head, and test the forward pass with dummy tensor shapes.

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
        
        # TODO 1: Load pre-trained DINOv2 backbone using timm
        # Backbone model key: 'vit_base_patch14_dinov2.lvd142m'
        # Hint: Set pretrained=True and num_classes=0 to load it as a feature extractor.
        
        self.backbone = None # TODO: Load model with pretrained=True and num_classes=0
        
        # TODO 2: Freeze ALL parameters in the backbone model
        # Hint: Loop through self.backbone.parameters() and set requires_grad to False
        # Write parameter freezing loop here:
        

        # Extract features dimension from backbone
        in_features = self.backbone.num_features
        print(f"DINOv2 Backbone backbone features shape: {in_features}")
        
        # TODO 3: Define a classification head
        # Use nn.Linear to map from in_features to num_classes
        
        self.head = None # TODO: Define linear head
        
    def forward(self, x):
        # TODO 4: Complete the forward pass logic
        # Extract features from backbone, then pass them through self.head
        # Hint: check how timm models process inputs. self.backbone(x) returns
        # the pooled features of shape (Batch_Size, num_features).
        
        features = None # TODO: extract features
        logits = None # TODO: pass features through head
        return logits

def main():
    # Number of target classes: 6 (Soybean diseases and healthy)
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
        # Input shape: (Batch Size, Channels, Height, Width)
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
