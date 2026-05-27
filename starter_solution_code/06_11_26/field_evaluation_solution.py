"""
Thursday, June 11 - Afternoon Session
Generalization and Domain-Shift Evaluation (Solution Code)

What this script is:
A reference solution for OOD accuracy evaluation.

Goal of this script:
Compute accuracies on validation vs. field test sets, print domain shift gaps, and plot comparative bar charts.

Why we are doing it (Student Context):
Models often fail in the real world due to domain shift (shadows, angles, camera quality). Evaluating this out-of-distribution accuracy helps us understand if the model generalizes.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

def main():
    val_targets = np.array([4, 4, 1, 1, 5, 5, 3, 3, 0, 2] * 10)
    val_preds   = np.array([4, 4, 1, 1, 5, 5, 3, 2, 0, 2] * 10) # 90% correct

    field_targets = np.array([4, 4, 1, 1, 5, 5, 3, 3, 0, 2] * 10)
    field_preds   = np.array([4, 1, 1, 4, 5, 3, 3, 2, 0, 3] * 10) # 60% correct

    # TODO 2 Solution: Calculate accuracies
    val_accuracy = np.mean(val_preds == val_targets)
    field_accuracy = np.mean(field_preds == field_targets)

    print("=========================================")
    print("Out-of-Distribution (OOD) Evaluation:")
    print("=========================================")
    print(f"Validation (In-Distribution) Accuracy: {val_accuracy * 100:.2f}%")
    print(f"Field Test (Out-of-Distribution) Accuracy: {field_accuracy * 100:.2f}%")
    print(f"Accuracy Gap (Domain Shift): {(val_accuracy - field_accuracy) * 100:.2f}%\n")

    # TODO 3 Solution: Comparative bar chart plotting
    labels = ['Validation Set (ID)', 'Field Test Set (OOD)']
    accuracies = [val_accuracy, field_accuracy]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, accuracies, color=["dodgerblue", "crimson"], width=0.4)
    
    # Customize plot
    plt.title("Model Performance Comparison: In-Distribution vs. Out-of-Distribution", fontsize=12, fontweight="bold")
    plt.ylabel("Classification Accuracy", fontsize=11)
    plt.ylim(0, 1.0)
    plt.grid(axis='y', linestyle=':', alpha=0.6)
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, height + 0.02, f"{height*100:.1f}%",
                 ha='center', va='bottom', fontsize=11, fontweight="bold")
                 
    # Save the evaluation plot
    os.makedirs("outputs", exist_ok=True)
    output_img = "outputs/field_vs_val_comparison.png"
    plt.savefig(output_img, dpi=300)
    print(f"Successfully generated and saved plot to '{output_img}'.")
    plt.show()

if __name__ == "__main__":
    main()
