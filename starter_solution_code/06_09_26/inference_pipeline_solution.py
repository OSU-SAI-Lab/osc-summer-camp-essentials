"""
Tuesday, June 9 - Morning Session
Image Prediction Pipeline (Solution Code)

What this script is:
A reference solution for preprocessing and single-image prediction.

Goal of this script:
Load model checkpoints, apply preprocessing transforms, add batch dimensions, run inference, and predict labels.

Why we are doing it (Student Context):
An AI model is only useful if it can run on new, unseen data in the real world. This pipeline bridges saved training checkpoints to real-world input images.
"""

import os
import torch
import torch.nn as nn
import timm
from PIL import Image
from torchvision import transforms

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

def preprocess_image(image_path):
    """
    Loads an image, converts it to RGB, applies resizing,
    converts to a tensor, and adds a batch dimension.
    """
    image = Image.open(image_path).convert("RGB")
    
    # TODO 1 Solution: Define preprocessing transforms
    preprocess = transforms.Compose([
        transforms.Resize((518, 518)),
        transforms.ToTensor(),
    ])
    
    img_tensor = preprocess(image)
    
    # TODO 2 Solution: Add batch dimension
    img_tensor = img_tensor.unsqueeze(0)
    
    return img_tensor

def predict(image_path, model_path, classes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize model
    model = SoybeanClassifier(num_classes=len(classes))
    
    # Load model weights if checkpoint exists
    if os.path.exists(model_path):
        print(f"Loading trained weights from checkpoint '{model_path}'...")
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and "model_state" in checkpoint:
            model.load_state_dict(checkpoint["model_state"])
        else:
            model.load_state_dict(checkpoint)
    else:
        print(f"Checkpoint '{model_path}' not found. Running with random weights for demo.")
        
    model = model.to(device)
    model.eval()
    
    # Preprocess image
    img_tensor = preprocess_image(image_path).to(device)
    
    # TODO 3 Solution: Run inference forward and apply softmax/argmax
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        
        # Pick the index with highest confidence
        pred_idx = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0, pred_idx].item()

    pred_class = classes[pred_idx]
    return pred_class, confidence

def main():
    classes = [
        'DicambaDamage', 
        'FrogEyeLeafSpot', 
        'GenericFeeding', 
        'InsectDamage', 
        'Soybeans', 
        'SuddenDeathSyndrome'
    ]
    
    MODEL_CHECKPOINT = "models/soybean_dinov2_head_model.pth"
    TEST_IMAGE = "outputs/sample_test_leaf.jpg"
    
    if not os.path.exists(TEST_IMAGE):
        import sys
        print(f"Error: Test image '{TEST_IMAGE}' not found.")
        print("Please place a sample test leaf image at that path to run inference.")
        sys.exit(1)
        
    try:
        pred_class, confidence = predict(TEST_IMAGE, MODEL_CHECKPOINT, classes)
        print("\n=========================================")
        print("Inference Pipeline Output:")
        print("=========================================")
        print(f"Test Leaf Image: {TEST_IMAGE}")
        print(f"Predicted Class: {pred_class}")
        print(f"Confidence Score: {confidence * 100:.2f}%")
        print("=========================================")
        
    except Exception as e:
        print(f"Inference failed: {e}")

if __name__ == "__main__":
    main()
