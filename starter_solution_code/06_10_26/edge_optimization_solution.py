"""
Wednesday, June 10 - Afternoon Session
Model Graph Optimization and Export (Solution Code)

What this script is:
A reference solution for edge optimization.

Goal of this script:
Trace model to TorchScript, export to ONNX, log file sizes, and profile average latency on CPU.

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
    
    device = torch.device("cpu")
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
    
    # Define dummy input
    dummy_input = torch.randn(1, 3, 518, 518).to(device)

    # TODO 1 Solution: Trace to TorchScript
    print("Compiling model to TorchScript (Tracing)...")
    ts_model = torch.jit.trace(model, dummy_input)
    
    os.makedirs("models", exist_ok=True)
    ts_output_path = "models/model_optimized.torchscript"
    torch.jit.save(ts_model, ts_output_path)
    print(f"  Saved TorchScript model to '{ts_output_path}'")

    # TODO 2 Solution: Export to ONNX
    print("\nExporting model to ONNX format...")
    onnx_output_path = "models/model_optimized.onnx"
    torch.onnx.export(
        model, 
        dummy_input, 
        onnx_output_path,
        input_names=['input'], 
        output_names=['output'],
        opset_version=11
    )
    print(f"  Saved ONNX model to '{onnx_output_path}'")
    
    # Save a temporary copy of baseline model weights to compare size
    baseline_path = "models/model_baseline.pth"
    torch.save(model.state_dict(), baseline_path)

    # TODO 3 Solution: File sizes comparison
    print("\n=========================================")
    print("Model Size Comparison:")
    print("=========================================")
    for path, name in [
        (baseline_path, "PyTorch State Dict (.pth)"),
        (ts_output_path, "TorchScript Compiled (.torchscript)"),
        (onnx_output_path, "ONNX Graph (.onnx)")
    ]:
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / (1024.0 * 1024.0)
            print(f"  - {name}: {size_mb:.2f} MB")
            
    # Clean up temp baseline file
    if os.path.exists(baseline_path):
        os.remove(baseline_path)
    
    # TODO 4 Solution: Benchmarking latency
    print("\nBenchmarking CPU latency (10 runs)...")
    
    # Baseline PyTorch time
    start_time = time.time()
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
    pytorch_duration = (time.time() - start_time) / 10.0
    print(f"  PyTorch average latency: {pytorch_duration*1000:.2f} ms per image")

    # TorchScript time
    start_time = time.time()
    with torch.no_grad():
        for _ in range(10):
            _ = ts_model(dummy_input)
    ts_duration = (time.time() - start_time) / 10.0
    print(f"  TorchScript average latency: {ts_duration*1000:.2f} ms per image")
    
    speedup = pytorch_duration / ts_duration
    print(f"  TorchScript Speedup Factor: {speedup:.2f}x")
    print("=========================================")

if __name__ == "__main__":
    main()
