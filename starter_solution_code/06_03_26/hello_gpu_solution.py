"""
Wednesday, June 3 - Morning Session
OSC GPU Environment Verification (Solution Code)

What this script is:
A reference solution for GPU node verification.

Goal of this script:
Verify CUDA GPU availability, allocate a test tensor on the GPU, and load the pre-trained DINOv2 model.

Why we are doing it (Student Context):
Before building complex AI systems, deep learning engineers must verify their environment has hardware acceleration (GPUs) working. This connects code to supercomputing resources.
"""

import sys
import torch
import timm

def verify_gpu_environment():
    print("=========================================")
    print("OSC GPU Environment Verification")
    print("=========================================")
    print(f"Python Version: {sys.version}")
    print(f"PyTorch Version: {torch.__version__}")

    # TODO 1 Solution: Check CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")

    if not cuda_available:
        print("WARNING: CUDA is not available. PyTorch is running on CPU.")
        device = torch.device("cpu")
    else:
        # TODO 2 Solution: Get count and name of primary GPU
        gpu_count = torch.cuda.device_count()
        device_name = torch.cuda.get_device_name(0)
        print(f"Number of GPUs: {gpu_count}")
        print(f"Primary GPU Device Name: {device_name}")
        device = torch.device("cuda:0")

    # TODO 3 Solution: Allocate tensor and move it to GPU
    print("\nAllocating test tensor on GPU...")
    test_tensor = torch.rand(3, 3, device=device)
    print(f"Tensor Location Device: {test_tensor.device}")
    print(f"Tensor Value:\n{test_tensor}")

    # TODO 4 Solution: Load pre-trained DINOv2 from timm
    print("\nLoading DINOv2 pre-trained backbone from timm registry...")
    try:
        model = timm.create_model('vit_base_patch14_dinov2.lvd142m', pretrained=True)
        model = model.to(device)
        print("DINOv2 Model loaded successfully!")
        print(f"Model Backbone Features: {model.num_features}")
        
        # Count trainable parameters
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total model parameters: {param_count:,}")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Tip: Make sure you have an internet connection or have cached the model weights.")

    print("\n=========================================")
    print("Verification completed successfully!")
    print("=========================================")

if __name__ == "__main__":
    verify_gpu_environment()
