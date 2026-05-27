"""
Friday, June 12 - Morning Session
Interactive Web Dashboard (Starter Code)

What this script is:
An interactive web application script.

Goal of this script:
Build a Gradio web application for crop diagnostics, allowing users to upload a leaf photo and receive predicted conditions, confidences, and recommendations.

Why we are doing it (Student Context):
Creating user interfaces allows non-technical users (like farmers or extension agents) to interact with and benefit from deep learning models, making the AI technology accessible and useful.
"""

import os
import torch
import torch.nn as nn
import timm
import gradio as gr
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

# Class list
CLASSES = [
    'DicambaDamage', 
    'FrogEyeLeafSpot', 
    'GenericFeeding', 
    'InsectDamage', 
    'Soybeans', 
    'SuddenDeathSyndrome'
]

MODEL_PATH = "models/soybean_dinov2_head_model.pth"

# Initialize and load model
device = torch.device("cpu")
model = SoybeanClassifier(num_classes=len(CLASSES))

if os.path.exists(MODEL_PATH):
    try:
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        if isinstance(checkpoint, dict) and "model_state" in checkpoint:
            model.load_state_dict(checkpoint["model_state"])
        else:
            model.load_state_dict(checkpoint)
    except Exception:
        pass
model.eval()

# Preprocessing transforms
preprocess = transforms.Compose([
    transforms.Resize((518, 518)),
    transforms.ToTensor(),
])

# Treatment recommendation mapper
def get_recommendation(predicted_class, confidence_score):
    if confidence_score < 0.60:
        return "Low confidence detection. Please retake the photo under better lighting, ensuring focus is on the leaf."
    if predicted_class == 'Soybeans':
        return "Crop is healthy. No treatment needed. Maintain standard watering and weed control."
    elif predicted_class == 'FrogEyeLeafSpot':
        return "Fungal spots (Frog Eye) detected. Monitor spot density. If spots cover >10% of the leaf canopy, apply a strobilurin fungicide."
    elif predicted_class == 'SuddenDeathSyndrome':
        return "Sudden Death Syndrome (Fusarium) detected. In-season chemical sprays are ineffective. Improve drainage; plan crop rotation next year."
    elif predicted_class == 'DicambaDamage':
        return "Chemical herbicide drift (Dicamba) detected. Document wind records, inspect buffer zones, and coordinate with neighboring farms."
    elif predicted_class in ['InsectDamage', 'GenericFeeding']:
        return "Active insect foliage feeding detected. Deploy sticky traps. Apply organic neem oil if feeding thresholds are exceeded."
    else:
        return "Unknown condition. Contact local extension office."

# TODO 1: Complete the prediction wrapper function for Gradio.
# Requirements:
#   1. Accept a PIL Image as input.
#   2. Apply preprocess transforms and add batch dimension using unsqueeze(0).
#   3. Run model forward pass under torch.no_grad().
#   4. Compute probabilities using torch.softmax, find predicted class index and confidence.
#   5. Call get_recommendation(class_name, confidence) to get the advice.
#   6. Return: class name, confidence text (e.g. "85.4%"), and recommendation text.

def analyze_leaf(pil_image):
    if pil_image is None:
        return "No image", "0.0%", "Please upload an image."
        
    # TODO: Preprocess and inference
    
    predicted_class = "Soybeans" # TODO: Replace
    confidence_val = 0.95 # TODO: Replace
    recommendation = get_recommendation(predicted_class, confidence_val)
    
    confidence_str = f"{confidence_val * 100:.1f}%"
    return predicted_class, confidence_str, recommendation

# TODO 2: Initialize the Gradio interface.
# Requirements:
#   - fn: Set to the analyze_leaf function
#   - inputs: Set to gr.Image(type="pil")
#   - outputs: Set to a list of three Textbox components:
#       - gr.Textbox(label="Predicted Crop Condition")
#       - gr.Textbox(label="Classification Confidence")
#       - gr.Textbox(label="Actionable Treatment Recommendation")
#   - title: Give it a premium dashboard title
#   - description: Add a brief descriptive tagline explaining the app

demo = None # TODO: Instantiate gr.Interface

# TODO 3: Launch the application
# Hint: Use demo.launch(share=True) to generate a public link for the closing ceremony!

if __name__ == "__main__":
    # Write launch code here:
    pass
