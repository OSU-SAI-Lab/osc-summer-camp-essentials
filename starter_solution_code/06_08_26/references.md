# Monday, June 8: Error Analysis & Hyperparameter Tuning

Today we enter Phase 2 of our project: evaluating our baseline Model v1 results and tuning hyperparameters to improve model performance.

---

##  Morning Session: Error Analysis & Metrics

### The Confusion Matrix
A classification model can achieve high overall accuracy but still fail critically on specific classes. To inspect this, we plot a **Confusion Matrix**:
*   **Rows**: Represent the true, ground-truth classes.
*   **Columns**: Represent the model's predicted classes.
*   **Diagonal Cells**: Represent correct predictions (True Positives).
*   **Off-Diagonal Cells**: Represent errors (e.g. predicting `SuddenDeathSyndrome` when the true label is `Soybeans` (healthy)).

### Key Metrics
1.  **Precision**: Of all samples the model predicted as class X, how many were actually class X?
    $$\text{Precision} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Positives}}$$
2.  **Recall**: Of all actual class X samples in the dataset, how many did the model find?
    $$\text{Recall} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Negatives}}$$
3.  **F1-Score**: The harmonic mean of Precision and Recall.

---

##  Afternoon Session: Hyperparameter Sweeping

When a baseline model confuses classes, we optimize performance by sweeping hyperparameters:
*   **Grid Search**: Trying every combination of a predefined list of parameters (e.g. trying learning rates `[1e-3, 1e-4]` with batch sizes `[16, 32]`).
*   **Logging Results**: To compare experiments, we log the hyperparameters and resulting test accuracy to a central registry (such as a JSON file).

```text
Experiment Sweep Loop:
 ├── Trial 1: LR = 1e-3, Accuracy = 82% ──> Log to JSON
 ├── Trial 2: LR = 1e-4, Accuracy = 89% ──> Log to JSON
 └── Trial 3: LR = 1e-5, Accuracy = 78% ──> Log to JSON
```

---

##  Curated Resources for Day 4
*   [Scikit-learn Classification Report](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html)
*   [Understanding the Confusion Matrix (Towards Data Science)](https://towardsdatascience.com/understanding-confusion-matrix-a9ad42dcfd62)
*   [Matplotlib Seaborn Heatmaps Guide](https://seaborn.pydata.org/generated/seaborn.heatmap.html)
