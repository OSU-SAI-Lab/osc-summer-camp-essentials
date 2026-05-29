"""
Friday, June 12 - Morning Session
Interactive Web Dashboard (Solution Code)

What this script is:
A reference solution for the Gradio web application.

Goal of this script:
Define predict wrappers, compile Gradio Interfaces with upload/textbox components, and launch sharing links.

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

# TODO 1 Solution: Prediction wrapper function
def analyze_leaf(pil_image):
    if pil_image is None:
        return "No image", "0.0%", "Please upload an image."
        
    # Preprocess
    img_tensor = preprocess(pil_image).unsqueeze(0).to(device)
    
    # Forward pass
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        pred_idx = torch.argmax(probabilities, dim=1).item()
        confidence_val = probabilities[0, pred_idx].item()
        
    predicted_class = CLASSES[pred_idx]
    recommendation = get_recommendation(predicted_class, confidence_val)
    
    confidence_str = f"{confidence_val * 100:.1f}%"
    return predicted_class, confidence_str, recommendation

# TODO 2 Solution: Initialize Gradio Interface
demo = gr.Interface(
    fn=analyze_leaf,
    inputs=gr.Image(type="pil", label="Upload Leaf Photo"),
    outputs=[
        gr.Textbox(label="Predicted Crop Condition"),
        gr.Textbox(label="Classification Confidence"),
        gr.Textbox(label="Actionable Treatment Recommendation")
    ],
    title=" Soybean Disease Diagnostics & Support System",
    description=("Upload an image of a soybean leaf to diagnose potential diseases (FLS, SDS, Herbicide Drift, Insects) "
                 "and retrieve farmer-friendly treatment recommendations immediately.")
)

# TODO 3 Solution: Launch interface
if __name__ == "__main__":
    demo.launch(share=True)
