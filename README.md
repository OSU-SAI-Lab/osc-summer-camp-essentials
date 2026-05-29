# AI in the Field: Visualizing Crop Health with Self-Supervised Learning (DINO)

Welcome to the **OSU Summer Camp 2026** curriculum repository! This codebase provides the starter templates and complete reference solutions for hands-on machine learning sessions running on the Ohio Supercomputer Center (OSC) cluster.

Students learn to train, optimize, evaluate, and deploy a self-supervised deep learning model ([DINOv2](https://github.com/facebookresearch/dinov2)) to identify crop disease from field-captured leaf images.

---

## Curriculum Schedule Overview | [Slides For The Whole Camp](https://buckeyemailosu-my.sharepoint.com/personal/kamath_62_buckeyemail_osu_edu/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fkamath%5F62%5Fbuckeyemail%5Fosu%5Fedu%2FDocuments%2Fosc%2Dsummer%2Dcamp&ga=1)

| Date | Morning Session (9:00 AM - 12:00 PM) | Afternoon Session (1:00 PM - 5:00 PM) | Key Resources | Slides |
| :--- | :--- | :--- | :--- | :--- |
| **Wed, June 3** | [OSC Foundations & Problem Framing](starter_solution_code/06_03_26/morning) | [Exploratory Data Analysis (EDA)](starter_solution_code/06_03_26/afternoon) | [Day 1 References](starter_solution_code/06_03_26/references.md) | [Slides](https://buckeyemailosu-my.sharepoint.com/:p:/g/personal/kamath_62_buckeyemail_osu_edu/IQCvILsY0tSxRacCDufh7xTCAVI9wzC4UH9jc9_zhhun3i4?e=ScOukk) |
| **Thu, June 4** | On-Site Field Data Collection | [Data Integration & Model Architecture](starter_solution_code/06_04_26/afternoon) | [Day 2 References](starter_solution_code/06_04_26/references.md) | [Slides](https://buckeyemailosu-my.sharepoint.com/:p:/g/personal/kamath_62_buckeyemail_osu_edu/IQCkxuC9ajhDQ7JO5neezVPEAUyKNHpnP1FcbDC4uJd9bAs?e=EBuXZd) |
| **Fri, June 5** | [Training Launch](starter_solution_code/06_05_26/morning) | [Monitoring & Optimization](starter_solution_code/06_05_26/afternoon) | [Day 3 References](starter_solution_code/06_05_26/references.md) | [Slides](https://buckeyemailosu-my.sharepoint.com/:p:/g/personal/kamath_62_buckeyemail_osu_edu/IQBxLXAefzOdQYehlHoFnq4bARlS3c3tchd-jFCZeHIKQks?e=UWOaT7) |
| **Mon, June 8** | [Error Analysis & Baseline Review](starter_solution_code/06_08_26/morning) | [Hyperparameter Tuning & Sweeps](starter_solution_code/06_08_26/afternoon) | [Day 4 References](starter_solution_code/06_08_26/references.md) | [Slides](https://buckeyemailosu-my.sharepoint.com/:p:/g/personal/kamath_62_buckeyemail_osu_edu/IQBnsp_rO6_uSKIazYFHjmG0AS4wafgQwBoa2gXtGsWqI2w?e=7xJFIg) |
| **Tue, June 9** | [Building the Inference Pipeline](starter_solution_code/06_09_26/morning) | [Actionable Agronomic Insights](starter_solution_code/06_09_26/afternoon) | [Day 5 References](starter_solution_code/06_09_26/references.md) | [Slides](https://buckeyemailosu-my.sharepoint.com/:p:/g/personal/kamath_62_buckeyemail_osu_edu/IQB48BbnNNTuQ5Qsv9irc5rLARTYn8wn0nhuo75U_vlYfoU?e=f6Lf7Z) |
| **Wed, June 10**| Self-Directed Prep (No Morning Session) | [Deployment Prep & Edge Optimization](starter_solution_code/06_10_26/afternoon) | [Day 6 References](starter_solution_code/06_10_26/references.md) | [Slides](https://buckeyemailosu-my.sharepoint.com/:p:/g/personal/kamath_62_buckeyemail_osu_edu/IQB4vurn4VH_SYffUETqkm3vAQh0uEeK4LYVhnytMRLQS-g?e=4HzX9I) |
| **Thu, June 11**| Live Field Testing & Inference | [Out-of-Distribution (OOD) Evaluation](starter_solution_code/06_11_26/afternoon) | [Day 7 References](starter_solution_code/06_11_26/references.md) | [Slides](https://buckeyemailosu-my.sharepoint.com/:p:/g/personal/kamath_62_buckeyemail_osu_edu/IQC74RqUaXRhTaMolQelunloASfFUPxNLfck5V1OcKTiRKA?e=1WlgGu) |
| **Fri, June 12**| [Gradio Dashboard Deployment](starter_solution_code/06_12_26/morning) | Final STEM Demos & Presentations | [Day 8 References](starter_solution_code/06_12_26/references.md) | [Slides](https://buckeyemailosu-my.sharepoint.com/:p:/g/personal/kamath_62_buckeyemail_osu_edu/IQBx8ErVUlZmT4U4e4bFmixYAZEZp2WFHXIAFjn1g1crjjo?e=xmhGDw) |

---

## Detailed Daily Sessions and Scripts

### Phase 1: Training (Week 1)

#### Day 1: Wednesday, June 3

* **[Slides](https://buckeyemailosu-my.sharepoint.com/:p:/r/personal/kamath_62_buckeyemail_osu_edu/_layouts/15/Doc.aspx?sourcedoc=%7B18BB20AF-D4D2-45B1-A702-0EE7E1EF14C2%7D&file=6-3-26.pptx&action=edit&mobileredirect=true)**
* **Morning Session: OSC Foundations and Problem Framing**
  * **Concepts**: Intro to High-Performance Computing (HPC), SLURM cluster queues, and visual identification of crop diseases.
  * **Hands-on Activity**: Accessing OSC, setting up environments, querying GPU devices, and loading a deep learning backbone network.
  * **Subfolder**: [06_03_26/morning](starter_solution_code/06_03_26/morning)
  * **Scripts**:
    * [hello_gpu_starter.py](starter_solution_code/06_03_26/morning/hello_gpu_starter.py) / [hello_gpu_solution.py](starter_solution_code/06_03_26/morning/hello_gpu_solution.py): Verifies PyTorch GPU access and loads the pre-trained DINOv2 model.
    * [hello_gpu_job.sh](starter_solution_code/06_03_26/morning/hello_gpu_job.sh) / [hello_gpu_job_solution.sh](starter_solution_code/06_03_26/morning/hello_gpu_job_solution.sh): SLURM batch job files to schedule tasks on GPU cluster nodes.
* **Afternoon Session: Exploratory Data Analysis (EDA) and DataLoaders**
  * **Concepts**: Image transformations, PyTorch `Dataset`/`DataLoader`, class imbalance distribution, and batch visualization.
  * **Hands-on Activity**: Writing transforms and augmentations for crop images.
  * **Subfolder**: [06_03_26/afternoon](starter_solution_code/06_03_26/afternoon)
  * **Scripts**:
    * [eda_dataloaders_starter.py](starter_solution_code/06_03_26/afternoon/eda_dataloaders_starter.py) / [eda_dataloaders_solution.py](starter_solution_code/06_03_26/afternoon/eda_dataloaders_solution.py): Preprocessing pipelines, class counts, and batch image plotting.

---

#### Day 2: Thursday, June 4

* **[Slides](https://buckeyemailosu-my.sharepoint.com/:p:/r/personal/kamath_62_buckeyemail_osu_edu/_layouts/15/Doc.aspx?sourcedoc=%7BBDE0C6A4-386A-4343-B24E-E6779ECD53C4%7D&file=6-4-26.pptx&action=edit&mobileredirect=true)**
* **Morning Session: Field Data Collection**
  * **Activity**: Out-of-lab session where students use mobile phones/edge cameras to collect disease samples under challenging lighting, angles, and background conditions.
* **Afternoon Session: Data Integration and Model Architecture**
  * **Concepts**: Transfer learning, freezing backbone parameters, classification layers, and shape checks.
  * **Hands-on Activity**: Integrating student field-captured images and constructing the DINOv2 classifier model.
  * **Subfolder**: [06_04_26/afternoon](starter_solution_code/06_04_26/afternoon)
  * **Scripts**:
    * [dino_architecture_starter.py](starter_solution_code/06_04_26/afternoon/dino_architecture_starter.py) / [dino_architecture_solution.py](starter_solution_code/06_04_26/afternoon/dino_architecture_solution.py): Builds a custom `SoybeanClassifier` PyTorch module using frozen DINOv2 feature maps, implementing output projections.

---

#### Day 3: Friday, June 5

* **[Slides](https://buckeyemailosu-my.sharepoint.com/:p:/r/personal/kamath_62_buckeyemail_osu_edu/_layouts/15/Doc.aspx?sourcedoc=%7B1E702D71-337F-419D-87A1-947A059EAE1B%7D&file=6-5-26.pptx&action=edit&mobileredirect=true)**
* **Morning Session: Training Launch**
  * **Concepts**: Separation of parameters from logic (YAML config files), PyTorch training loops, multi-stage fine-tuning (head vs. full backbone block).
  * **Hands-on Activity**: Decoupling settings and firing up a long-running batch job on GPU nodes.
  * **Subfolder**: [06_05_26/morning](starter_solution_code/06_05_26/morning)
  * **Scripts**:
    * [config_starter.yaml](starter_solution_code/06_05_26/morning/config_starter.yaml) / [config_solution.yaml](starter_solution_code/06_05_26/morning/config_solution.yaml): Hyperparameter definitions (batch size, learning rate, weight decay).
    * [train_starter.py](starter_solution_code/06_05_26/morning/train_starter.py) / [train_solution.py](starter_solution_code/06_05_26/morning/train_solution.py): Multi-phase training script (header training + backbone fine-tuning) with model checkpoint saving.
    * [train_job_starter.sh](starter_solution_code/06_05_26/morning/train_job_starter.sh) / [train_job_solution.sh](starter_solution_code/06_05_26/morning/train_job_solution.sh): Cluster batch runner.
* **Afternoon Session: Monitoring and Optimization**
  * **Concepts**: SLURM logging, regex log parsing, loss/accuracy curve plotting, and debugging divergence.
  * **Hands-on Activity**: Monitoring runs in progress, parsing and plotting losses.
  * **Subfolder**: [06_05_26/afternoon](starter_solution_code/06_05_26/afternoon)
  * **Scripts**:
    * [plot_logs_starter.py](starter_solution_code/06_05_26/afternoon/plot_logs_starter.py) / [plot_logs_solution.py](starter_solution_code/06_05_26/afternoon/plot_logs_solution.py): Regex utility to scan the newest log files and export graphs of performance metrics.

---

### Phase 2: Project Work (Week 2)

#### Day 4: Monday, June 8

* **[Slides](https://buckeyemailosu-my.sharepoint.com/:p:/r/personal/kamath_62_buckeyemail_osu_edu/_layouts/15/Doc.aspx?sourcedoc=%7BEB9FB267-AF3B-48EE-A21A-CD81478E61B4%7D&file=6-8-26.pptx&action=edit&mobileredirect=true)**
* **Morning Session: Error Analysis and Baseline Review**
  * **Concepts**: Classification report metrics (Precision, Recall, F1), confusion matrices, and diagnosing model bias.
  * **Hands-on Activity**: Visualizing confusion matrices to pinpoint overlapping features (e.g., Healthy vs. Herbicide Damage).
  * **Subfolder**: [06_08_26/morning](starter_solution_code/06_08_26/morning)
  * **Scripts**:
    * [error_analysis_starter.py](starter_solution_code/06_08_26/morning/error_analysis_starter.py) / [error_analysis_solution.py](starter_solution_code/06_08_26/morning/error_analysis_solution.py): Generates performance reports and confusion matrix heatmap figures.
    * [error_analysis_job_starter.sh](starter_solution_code/06_08_26/morning/error_analysis_job_starter.sh) / [error_analysis_job_solution.sh](starter_solution_code/06_08_26/morning/error_analysis_job_solution.sh): Runs the metric calculations on the GPU node.
* **Afternoon Session: Hyperparameter Sweeping**
  * **Concepts**: Hyperparameter exploration, metadata export to JSON, and parallel trial comparisons.
  * **Hands-on Activity**: Running a grid sweep of learning rates and charting comparative validation metrics.
  * **Subfolder**: [06_08_26/afternoon](starter_solution_code/06_08_26/afternoon)
  * **Scripts**:
    * [hyperparameter_tuning_starter.py](starter_solution_code/06_08_26/afternoon/hyperparameter_tuning_starter.py) / [hyperparameter_tuning_solution.py](starter_solution_code/06_08_26/afternoon/hyperparameter_tuning_solution.py): Coordinates multiple runs, stores configurations and results, and plots a comparison bar chart.
    * [hyperparameter_tuning_job_starter.sh](starter_solution_code/06_08_26/afternoon/hyperparameter_tuning_job_starter.sh) / [hyperparameter_tuning_job_solution.sh](starter_solution_code/06_08_26/afternoon/hyperparameter_tuning_job_solution.sh): Submits the hyperparameter sweeps.

---

#### Day 5: Tuesday, June 9

* **[Slides](https://buckeyemailosu-my.sharepoint.com/:p:/r/personal/kamath_62_buckeyemail_osu_edu/_layouts/15/Doc.aspx?sourcedoc=%7BE716F078-D434-43EE-942C-BFD8AB739ACB%7D&file=6-9-26.pptx&action=edit&mobileredirect=true)**
* **Morning Session: Inference Pipeline**
  * **Concepts**: Production pipeline design, loading model weights, inference under `torch.no_grad()`, and mapping classification outputs.
  * **Hands-on Activity**: Building a clean python script to predict crop health labels on arbitrary image inputs.
  * **Subfolder**: [06_09_26/morning](starter_solution_code/06_09_26/morning)
  * **Scripts**:
    * [inference_pipeline_starter.py](starter_solution_code/06_09_26/morning/inference_pipeline_starter.py) / [inference_pipeline_solution.py](starter_solution_code/06_09_26/morning/inference_pipeline_solution.py): Image loader and model predictor return structure.
* **Afternoon Session: Actionable Insights**
  * **Concepts**: Agricultural treatment mappings, safety confidence thresholds, and farm decision dashboards.
  * **Hands-on Activity**: Linking crop disease predictions to actionable agronomic treatments.
  * **Subfolder**: [06_09_26/afternoon](starter_solution_code/06_09_26/afternoon)
  * **Scripts**:
    * [treatment_recommender_starter.py](starter_solution_code/06_09_26/afternoon/treatment_recommender_starter.py) / [treatment_recommender_solution.py](starter_solution_code/06_09_26/afternoon/treatment_recommender_solution.py): Maps classification names to specific treatment files and flags low-certainty predictions.

---

#### Day 6: Wednesday, June 10

* **[Slides](https://buckeyemailosu-my.sharepoint.com/:p:/r/personal/kamath_62_buckeyemail_osu_edu/_layouts/15/Doc.aspx?sourcedoc=%7BE7EABE78-51E1-49FF-87DF-5044EA926DEF%7D&file=6-10-26.pptx&action=edit&mobileredirect=true)**
* **Afternoon Session: Deployment Prep and Edge Optimization**
  * **Concepts**: Graph serialization, compilation frameworks (TorchScript vs. ONNX), binary file footprint, and hardware runtime benchmarks.
  * **Hands-on Activity**: Compiling the trained models to optimized graphs and comparing inference latencies.
  * **Subfolder**: [06_10_26/afternoon](starter_solution_code/06_10_26/afternoon)
  * **Scripts**:
    * [edge_optimization_starter.py](starter_solution_code/06_10_26/afternoon/edge_optimization_starter.py) / [edge_optimization_solution.py](starter_solution_code/06_10_26/afternoon/edge_optimization_solution.py): Traces models to TorchScript and ONNX, benchmarking speedups.
    * [edge_optimization_job_starter.sh](starter_solution_code/06_10_26/afternoon/edge_optimization_job_starter.sh) / [edge_optimization_job_solution.sh](starter_solution_code/06_10_26/afternoon/edge_optimization_job_solution.sh): Runs serializations and latency benchmarks on the cluster hardware.

---

#### Day 7: Thursday, June 11

* **[Slides](https://buckeyemailosu-my.sharepoint.com/:p:/r/personal/kamath_62_buckeyemail_osu_edu/_layouts/15/Doc.aspx?sourcedoc=%7B941AE1BB-7469-4D61-A328-9507A5BA7968%7D&file=6-11-26.pptx&action=edit&mobileredirect=true)**
* **Afternoon Session: Out-of-Distribution (OOD) Evaluation**
  * **Concepts**: Domain shift, out-of-distribution (OOD) testing, and generalization gap profiling.
  * **Hands-on Activity**: Benchmarking the differences between clean, curated validation sets and messy, live phone-captured images.
  * **Subfolder**: [06_11_26/afternoon](starter_solution_code/06_11_26/afternoon)
  * **Scripts**:
    * [field_evaluation_starter.py](starter_solution_code/06_11_26/afternoon/field_evaluation_starter.py) / [field_evaluation_solution.py](starter_solution_code/06_11_26/afternoon/field_evaluation_solution.py): Evaluates models on raw test datasets and plots accuracy comparisons.
    * [field_evaluation_job_starter.sh](starter_solution_code/06_11_26/afternoon/field_evaluation_job_starter.sh) / [field_evaluation_job_solution.sh](starter_solution_code/06_11_26/afternoon/field_evaluation_job_solution.sh): Triggers OOD model evaluations.

---

#### Day 8: Friday, June 12

* **[Slides](https://buckeyemailosu-my.sharepoint.com/:p:/r/personal/kamath_62_buckeyemail_osu_edu/_layouts/15/Doc.aspx?sourcedoc=%7BD54AF071-5652-4F66-8538-7B86C59A2C58%7D&file=6-12-26.pptx&action=edit&mobileredirect=true)**
* **Morning Session: Final Refinement and Gradio Dashboard**
  * **Concepts**: Graphical Web User Interfaces (GUIs), interactive components, and public sharing.
  * **Hands-on Activity**: Combining model prediction, treatment mapping, and an intuitive user interface into a real-time web application.
  * **Subfolder**: [06_12_26/morning](starter_solution_code/06_12_26/morning)
  * **Scripts**:
    * [dashboard_starter.py](starter_solution_code/06_12_26/morning/dashboard_starter.py) / [dashboard_solution.py](starter_solution_code/06_12_26/morning/dashboard_solution.py): Creates and launches the interactive Gradio web server app.
* **Afternoon Session: Closing STEM Demos**
  * **Activity**: Closing presentations demonstrating functional web applications diagnosing soybean crop diseases.
