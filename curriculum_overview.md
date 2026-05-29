# Curriculum Overview: AI in the Field (OSU Summer Camp 2026)

This document provides a day-by-day reference of the summer camp schedule, the hands-on sessions, and the technical details of the scripts students will write and run.

---

## Phase 1: Training (Week 1)

### Day 1: Wednesday, June 3

* **Morning Session: OSC Foundations and Problem Framing**
  * **Concept**: Intro to High-Performance Computing (HPC), cluster queues (SLURM), and crop disease visual identification.
  * **Hands-on**: Verify GPU resources and load pre-trained deep learning backbone networks on OSC.
  * **Scripts**:
    * hello_gpu_starter.py / hello_gpu_solution.py: Verifies PyTorch can access GPU, queries device count/names, allocates a test tensor, and loads a DINOv2 model.
    * hello_gpu_job.sh / hello_gpu_job_solution.sh: SLURM batch scripts to submit the verification task to the GPU node queue.
* **Afternoon Session: Exploratory Data Analysis (EDA) and DataLoaders**
  * **Concept**: Data preprocessing, data augmentation, class imbalance, and PyTorch datasets.
  * **Hands-on**: Build crop image augmentations and data loaders.
  * **Scripts**:
    * eda_dataloaders_starter.py / eda_dataloaders_solution.py: Composes image transforms (Resize, HorizontalFlip, RandomRotation), loads datasets using ImageFolder, calculates class distributions, and plots batch images.

---

### Day 2: Thursday, June 4

* **Morning Session: Field Data Collection**
  * **On-Site Activity**: Students go out to fields with edge cameras/mobile devices to capture real soybean leaf disease crop data under varying lights, angles, and backgrounds.
* **Afternoon Session: Data Integration and Model Architecture**
  * **Concept**: Transfer learning, frozen backbones, and classification heads.
  * **Hands-on**: Integrate new field data and build the DINOv2-based Classifier architecture.
  * **Scripts**:
    * dino_architecture_starter.py / dino_architecture_solution.py: Implements a custom PyTorch model wrapper (SoybeanClassifier) that wraps a pre-trained feature-extracting DINOv2 model (num_classes=0), freezes the backbone weights, adds a classification head (nn.Linear), and validates forward pass tensor shapes.

---

### Day 3: Friday, June 5

* **Morning Session: Training Launch**
  * **Concept**: Configuration decoupling, hyperparameter options, and batch job submission.
  * **Hands-on**: Decouple parameters into YAML configurations and launch a multi-stage training run on the GPU cluster.
  * **Scripts**:
    * config_starter.yaml / config_solution.yaml: Hyperparameter settings (learning rates, batch sizes, epochs).
    * train_starter.py / train_solution.py: Full training script. Splits data, trains the classification head, unfreezes the last ViT block, fine-tunes, and saves checkpoints to .pth files.
    * train_job_starter.sh / train_job_solution.sh: Submits the GPU training job to the cluster.
* **Afternoon Session: Monitoring and Optimization**
  * **Concept**: Monitoring loss curves and troubleshooting training divergence.
  * **Hands-on**: Parse cluster output log files and plot real-time loss curves.
  * **Scripts**:
    * plot_logs_starter.py / plot_logs_solution.py: Automatically finds the newest SLURM log output, parses loss metrics using Regular Expressions, and generates sequential loss curve plots.

---

## Phase 2: Project Work (Week 2)

### Day 4: Monday, June 8

* **Morning Session: Error Analysis and Baseline Review**
  * **Concept**: Performance metrics (Precision, Recall, F1-Score) and confusion heatmaps.
  * **Hands-on**: Identify confused classes (e.g. Herbicide Damage vs Healthy) using visual statistics.
  * **Scripts**:
    * error_analysis_starter.py / error_analysis_solution.py: Computes precision, recall, and f1-scores, and generates an annotated confusion matrix heatmap.
* **Afternoon Session: Hyperparameter Sweeping**
  * **Concept**: Search spaces, logging results, and comparative chart analysis.
  * **Hands-on**: Run a simulated learning rate sweep and save trial results to JSON.
  * **Scripts**:
    * hyperparameter_tuning_starter.py / hyperparameter_tuning_solution.py: Evaluates multiple learning rate settings, logs outcomes to JSON, and plots a bar chart comparing validation accuracy.

---

### Day 5: Tuesday, June 9

* **Morning Session: Inference Pipeline**
  * **Concept**: Deployment processing, forward passes under no_grad(), and softmax probability scoring.
  * **Hands-on**: Load saved weights and run predictions on raw leaf images.
  * **Scripts**:
    * inference_pipeline_starter.py / inference_pipeline_solution.py: Loads the saved model weights, preprocesses individual PIL images, runs forward passes, and extracts class indices and confidence scores.
* **Afternoon Session: Actionable Insights**
  * **Concept**: Decision support systems, agricultural domain mapping, and uncertainty thresholds.
  * **Hands-on**: Map model outputs to actionable farmer-facing recommendations.
  * **Scripts**:
    * treatment_recommender_starter.py / treatment_recommender_solution.py: Maps predictions to specific agronomic treatments (canopy sprays, drainage improvements, etc.) and flags low-confidence predictions (<60%) for recapturing.

---

### Day 6: Wednesday, June 10

* **Afternoon Session: Deployment Prep and Edge Optimization**
  * **Concept**: Model compression, graph serialization, ONNX runtime, and latency benchmarking.
  * **Hands-on**: Compile the model to ONNX and TorchScript, check file size reductions, and measure inference speedups.
  * **Scripts**:
    * edge_optimization_starter.py / edge_optimization_solution.py: Traces model graphs, exports to ONNX, logs file sizes, and profiles average latency on CPU over multiple forward pass runs.

---

### Day 7: Thursday, June 11

* **Afternoon Session: Out-of-Distribution (OOD) Evaluation**
  * **Concept**: Domain shift, out-of-distribution data, and generalization gaps.
  * **Hands-on**: Evaluate performance on validation (in-distribution) data vs. newly captured mobile phone field data.
  * **Scripts**:
    * field_evaluation_starter.py / field_evaluation_solution.py: Calculates validation accuracy vs. field accuracy and plots a comparative bar chart displaying the generalization gap.

---

### Day 8: Friday, June 12

* **Morning Session: Final Refinement and Gradio Dashboard**
  * **Concept**: Graphical user interfaces (GUI), web apps, and closing ceremony product demo preparation.
  * **Hands-on**: Build a full web application for farmers to diagnose crop disease inputs instantly.
  * **Scripts**:
    * dashboard_starter.py / dashboard_solution.py: Instantiates a Gradio web application with textboxes, image upload widgets, and maps predictions to recommendations, launching with public sharing URLs.
