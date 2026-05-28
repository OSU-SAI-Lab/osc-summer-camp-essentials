# Thursday, June 11: Field Testing & Live Inference Evaluation

Today we test our optimized models in the field using freshly captured images, and compile results to compare training/validation accuracy vs. real-world field testing performance.

---

## Morning Session: Field Testing (Slot 2)

In the morning session, groups will run their inference pipelines on edge devices using raw images captured directly in the fields today.

---

## Afternoon Session: Domain Shift & OOD Evaluation

### Why Models Fail in the Field: Domain Shift

It is common for a machine learning model to achieve **95% accuracy** on its validation dataset during training, but drop to **60-70% accuracy** when tested in the field. This drop is caused by **Domain Shift** (or Out-Of-Distribution data):

1. **Sensor Differences**: The training dataset was captured with professional cameras or specific edge devices, while the new test images were captured using different smartphone sensors.
2. **Environmental Variations**: Variations in soil type, background weeds, sky reflections, leaf shadows, and wetness from morning dew can confuse the model.
3. **Geographical/Temporal Shift**: The training leaves were collected in a laboratory or early in the season, while the field test leaves are older or grown in a different field.

### Comparing In-Distribution vs. Out-of-Distribution Accuracy

To understand how robust our model is, we compile performance metrics on:

* **In-Distribution (ID) Set**: The original training/validation test split.
* **Out-of-Distribution (OOD)**: Evaluate on the field images collected by students.
* **Analyze Drop**: Quantify the performance drop to understand how much the model struggles with real-world noise.

### Running on the HPC

Evaluating two full datasets requires GPU acceleration. Submit this task as a batch job:

```bash
sbatch field_evaluation_job_solution.sh
```

We plot these accuracies side-by-side in a comparative bar chart. A narrow gap indicates a highly robust model; a wide gap indicates overfitting or extreme sensitivity to environmental noise.

---

## Curated Resources for Day 7

* [Understanding Domain Adaptation and Domain Shift](https://towardsdatascience.com/understanding-domain-adaptation-5f5d6f35b62b)
* [Out-of-Distribution Generalization in Deep Learning](https://arxiv.org/abs/2108.13624)
* [Matplotlib Comparative Bar Charts Guide](https://matplotlib.org/stable/gallery/lines_bars_and_markers/barchart.html)
