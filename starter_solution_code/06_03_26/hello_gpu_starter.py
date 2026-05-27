"""
Wednesday, June 3 - Morning Session
OSC GPU Environment Verification (Starter Code)

What this script is:
A GPU node verification script.

Goal of this script:
Check PyTorch and Python versions, verify CUDA GPU availability, allocate a test tensor on the GPU, and load the pre-trained DINOv2 model.

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

    # TODO 1: Check if CUDA (GPU support) is available in PyTorch
    # Hint: Use torch.cuda.is_available()

    cuda_available = False # TODO: Replace with check
    print(f"CUDA Available: {cuda_available}")

    if not cuda_available:
        print("WARNING: CUDA is not available. PyTorch is running on CPU.")
        device = torch.device("cpu")
    else:
        # TODO 2: Get the number of available GPUs and print the name of device 0.
        # Hint: Use torch.cuda.device_count() and torch.cuda.get_device_name(0)
        
        gpu_count = 0 # TODO: Replace with check
        device_name = "None" # TODO: Replace with check
        print(f"Number of GPUs: {gpu_count}")
        print(f"Primary GPU Device Name: {device_name}")
        device = torch.device("cuda:0")

    # TODO 3: Create a random 3x3 tensor and move it to the device (GPU)
    # Hint: Use torch.rand(3, 3).to(device) or torch.rand(3, 3, device=device)
    
    print("\nAllocating test tensor on GPU...")
    test_tensor = None # TODO: Replace with tensor allocation
    print(f"Tensor Location Device: {test_tensor.device if test_tensor is not None else 'None'}")
    print(f"Tensor Value:\n{test_tensor}")

    # TODO 4: Load a vision model backbone from timm and print its structure
    # Load 'vit_base_patch14_dinov2.lvd142m' with pretrained=True
    
    print("\nLoading DINOv2 pre-trained backbone from timm registry...")
    try:
        model = None # TODO: Load model using timm.create_model
        if model is not None:
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
