# Wednesday, June 10: Edge Optimization & Deployment Prep

Today we prepare our model for deployment on resource-constrained edge devices (like smartphones, raspberry pis, or drones) by converting our PyTorch model to optimized formats (TorchScript and ONNX).

---

## Afternoon Session: Edge Deployment Optimization

### Why Optimize?

Deep learning models can contain hundreds of millions of parameters. Running these on edge devices presents several challenges:

1. **Size**: A large model (e.g. 300MB) takes up too much memory.
2. **Latency**: Forward passes can take seconds on edge CPUs, making real-time field diagnostics impossible.
3. **Dependency**: Deploying raw PyTorch code requires installing a full Python environment and the large `torch` library on the device.

### Optimization Formats

#### 1. TorchScript

TorchScript is PyTorch's native compiler. It creates a serialized representation of your model that can be loaded and executed directly in a **C++ application** (without any Python interpreter needed).

* **Tracing**: We pass a dummy input tensor through the model, and TorchScript records (traces) all operators executed.
* `torch.jit.trace(model, dummy_input)`: Generates the TorchScript compiled representation.

#### 2. ONNX (Open Neural Network Exchange)

ONNX is an open ecosystem that allows AI developers to move models between different frameworks (e.g., training in PyTorch and running in TensorFlow or OpenVINO).

* **ONNX Runtime**: A high-performance execution engine designed to run ONNX models with hardware acceleration on mobile and edge CPUs.
* `torch.onnx.export(model, dummy_input, "model.onnx")`: Compiles the PyTorch model graph into the ONNX format.

### Running on the HPC

Since compiling models requires tracing heavy forward passes, we execute this step on the HPC compute nodes.

```bash
# Submit Edge Optimization
sbatch edge_optimization_job_solution.sh
```

---

## Curated Resources for Day 6

* [Introduction to TorchScript](https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html)
* [PyTorch ONNX Export Documentation](https://pytorch.org/docs/stable/onnx.html)
* [ONNX Runtime Quick Start Guide](https://onnxruntime.ai/docs/get-started/with-python.html)
