"""
Wednesday, June 10 - Afternoon Session
Model Graph Optimization and Export (Starter Code)

What this script is:
An edge deployment optimization and benchmarking script.

Goal of this script:
Export PyTorch models to TorchScript and ONNX formats, compare file sizes, and profile CPU latency.

Why we are doing it (Student Context):
Deep learning models are heavy. For edge deployment (like mobile apps used in the field), we must compress models and compile graphs to run fast on lightweight CPU processors without network connections.
"""

import os
import time
import torch
import torch.nn as nn
import timm

class SoybeanClassifier(nn.Module):
    def __init__(self, num_classes=6):
        super(SoybeanClassifier, self).__init__()
        # Use resnet18 as a lightweight local model for fast export tests
        self.backbone = timm.create_model('resnet18', pretrained=False, num_classes=0)
        in_features = self.backbone.num_features
        self.head = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        features = self.backbone(x)
        logits = self.head(features)
        return logits

def main():
    NUM_CLASSES = 6
    MODEL_PATH = "models/soybean_dinov2_head_model.pth"
    
    device = torch.device("cpu") # Benchmarking on CPU (typical for mobile devices)
    print(f"Running edge optimizations on device: {device}\n")
    
    # Initialize baseline model
    model = SoybeanClassifier(num_classes=NUM_CLASSES)
    if os.path.exists(MODEL_PATH):
        try:
            checkpoint = torch.load(MODEL_PATH, map_location=device)
            if isinstance(checkpoint, dict) and "model_state" in checkpoint:
                model.load_state_dict(checkpoint["model_state"])
            else:
                model.load_state_dict(checkpoint)
        except Exception:
            pass
            
    model = model.to(device).eval()
    
    # Define dummy input matching DINOv2 dimensions
    # Shape: (1, Channels, Height, Width)
    dummy_input = torch.randn(1, 3, 518, 518).to(device)

    # TODO 1: Compile the model to TorchScript using Tracing
    # Hint: Use torch.jit.trace(model, dummy_input)
    
    print("Compiling model to TorchScript (Tracing)...")
    ts_model = None # TODO: trace model
    
    os.makedirs("models", exist_ok=True)
    ts_output_path = "models/model_optimized.torchscript"
    # Save the TorchScript model
    if ts_model is not None:
        torch.jit.save(ts_model, ts_output_path)
        print(f"  Saved TorchScript model to '{ts_output_path}'")

    # TODO 2: Export the model to ONNX format
    # Hint: Use torch.onnx.export(model, dummy_input, output_path,
    #                            input_names=['input'], output_names=['output'],
    #                            opset_version=11)
    
    print("\nExporting model to ONNX format...")
    onnx_output_path = "models/model_optimized.onnx"
    # Write ONNX export code here:
    
    
    # TODO 3: Save baseline model state_dict to compare size, then measure and print model file sizes.
    # Hint: Save baseline using torch.save(model.state_dict(), "models/model_baseline.pth").
    # Hint: Use os.path.getsize(path) / (1024 * 1024) to get size in Megabytes (MB) for:
    #       - "model_baseline.pth"
    #       - "model_optimized.torchscript"
    #       - "model_optimized.onnx"
    # Remember to clean up "model_baseline.pth" using os.remove() after size printing.
    
    print("\n=========================================")
    print("Model Size Comparison:")
    print("=========================================")
    # Print file sizes here:
    
    
    # TODO 4: Profile CPU latency performance (Run 10 forward passes)
    # Compare forward pass latency of PyTorch model vs. TorchScript model.
    
    print("\nBenchmarking CPU latency (10 runs)...")
    
    # Baseline PyTorch time
    start_time = time.time()
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
    pytorch_duration = (time.time() - start_time) / 10.0
    print(f"  PyTorch average latency: {pytorch_duration*1000:.2f} ms per image")

    # TorchScript time
    if ts_model is not None:
        start_time = time.time()
        with torch.no_grad():
            for _ in range(10):
                # TODO: Run forward pass on ts_model
                
                pass
        ts_duration = (time.time() - start_time) / 10.0
        print(f"  TorchScript average latency: {ts_duration*1000:.2f} ms per image")
        
        speedup = pytorch_duration / ts_duration
        print(f"  TorchScript Speedup Factor: {speedup:.2f}x")
    print("=========================================")

if __name__ == "__main__":
    main()
