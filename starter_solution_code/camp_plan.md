#  AI in the Field: Visualizing Crop Health with Self-Supervised Learning (DINO)

Welcome to the **OSU Summer Camp 2026**! Below is the comprehensive daily schedule and plan for our two-week program.

---

## Schedule Overview

| Date | Phase | Morning Session | Afternoon Session |
| :--- | :--- | :--- | :--- |
| **Wed, June 3** | Phase 1: Training | OSC Foundations & Problem Framing | Exploratory Data Analysis (EDA) |
| **Thu, June 4** | Phase 1: Training | Field Data Collection (Slot 1) | Data Integration & Model Architecture |
| **Fri, June 5** | Phase 1: Training | Training Launch | Monitoring & Optimization |
| **Mon, June 8** | Phase 2: Project Work | Error Analysis & Baseline Review | Hyperparameter Tuning & Experimentation |
| **Tue, June 9** | Phase 2: Project Work | Building the Inference Pipeline | Actionable Insights |
| **Wed, June 10** | Phase 2: Project Work | *(Self-Directed Prep)* | Deployment Prep & System Optimization |
| **Thu, June 11** | Phase 2: Project Work | Field Testing & Live Inference (Slot 2) | Results Compilation & Presentation Practice |
| **Fri, June 12** | Phase 2: Project Work | Final Refinement | Presentation Practice (STEM Closing) |

---

##  Detailed Daily Agenda

###  Phase 1: Project Training (Week 1)

####  Wednesday, June 3

* ** 9:00am – 12:30pm (3.5 hours): OSC Foundations & Problem Framing**
  * **HPC Environment:** Log in to OnDemand, navigate filesystems, and module load required software.
  * Introduction to Soybean diseases and the computer vision challenges of FLS, SDS, and BBP.
  * >  **Layer 4 Hands-on:** Execute a "Hello World" training script on a GPU node to verify the environment.
* ** 1:30pm – 3:00pm (1.5 hours): Exploratory Data Analysis (EDA)**
  * Inspect existing datasets for class imbalance and noise.
  * >  **Layer 2 Hands-on:** Build PyTorch DataLoaders and visualize data augmentations.

---

####  Thursday, June 4

* ** 9:00am – 11:45am (2.75 hours): Field Data Collection (First Field Slot)**
  * **On-Site Activity:** Students use edge devices to capture live images of crops in the field.
  * Focus on capturing diverse lighting, angles, and disease examples to supplement the training set.
* ** 2:45pm – 5:00pm (2.15 hours): Data Integration & Model Architecture**
  * Upload field data to OSC and merge with the primary dataset.
  * >  **Layer 3 Hands-on:** Implement the DINO (Teacher-Student) head and classification layer logic.

---

####  Friday, June 5

* ** 9:00am – 9:45am (0.75 hours): Training Launch**
  * Configure hyperparameters and submit the DINO training job to the OSC batch queue.
* ** 1:00pm – 5:00pm (4 hours): Monitoring & Optimization**
  * Track performance metrics like Accuracy and Per-class Recall.
  * >  **Layer 1 Hands-on:** Write scripts to parse training logs and plot real-time loss curves.

---

###  Phase 2: Project Work (Week 2)

####  Monday, June 8

* ** 9:00am – 12:00pm (3 hours): Error Analysis & Baseline Review**
  * Evaluate Model v1 results; identify "confused" classes (e.g., BBP vs. Healthy).
* ** 12:45pm – 5:00pm (4.25 hours): Hyperparameter Tuning & Experimentation**
  * Adjust learning rates and augmentation strategies to improve robustness.

---

####  Tuesday, June 9

* ** 9:00am – 12:00pm (3 hours): Building the Inference Pipeline**
  * Develop a script to load the trained `.pth` model and process raw image inputs.
* ** 2:30pm – 5:00pm (2.5 hours): Actionable Insights**
  * Map model confidence scores to farmer-friendly "Treatment Recommendations".

---

####  Wednesday, June 10

* ** 1:00pm – 5:00pm (4 hours): Deployment Prep & System Optimization**
  * Finalize the inference script for edge device compatibility.
  * Prepare the system for tomorrow's live field testing.

---

####  Thursday, June 11

* ** 9:00am – 12:00pm (3 hours): Field Testing & Live Inference (Second Field Slot)**
  * **On-Site Activity:** Test the working systems in the field using live images.
  * Run inference on newly captured test data to validate real-world performance.
* ** 3:15pm – 5:00pm (1.75 hours): Results Compilation & Presentation Practice**
  * Document field performance vs. training performance.
  * Draft technical summaries for the closing ceremony.

---

####  Friday, June 12

* ** 9:00am – 12:00pm (3 hours): Final Refinement**
  * Polish the "Project Folder" (Layers 1-4) and finalize demo visualizations.
* ** 1:00pm – 5:00pm (4 hours): Presentation Practice**
  * Final group rehearsal for the STEM Institute Closing Ceremony.
