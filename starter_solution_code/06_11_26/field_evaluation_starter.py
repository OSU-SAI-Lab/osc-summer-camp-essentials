"""
Thursday, June 11 - Afternoon Session
Generalization and Domain-Shift Evaluation (Starter Code)

What this script is:
An out-of-distribution (OOD) accuracy evaluation script.

Goal of this script:
Compare model accuracy on validation (in-distribution) data vs. newly captured mobile phone (out-of-distribution) data, and plot comparative bar charts.

Why we are doing it (Student Context):
Models often fail in the real world due to domain shift (shadows, angles, camera quality). Evaluating this out-of-distribution accuracy helps us understand if the model generalizes.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

def main():
    # TODO 1: Define validation predictions and field test predictions.
    # We will simulate the evaluations here.
    # Validation targets and predictions:
    
    val_targets = np.array([4, 4, 1, 1, 5, 5, 3, 3, 0, 2] * 10)
    val_preds   = np.array([4, 4, 1, 1, 5, 5, 3, 2, 0, 2] * 10) # 90% correct

    # Field test targets and predictions (collected on new mobile phones today):
    # Note: Confusions are higher due to shadows and different camera angles.
    field_targets = np.array([4, 4, 1, 1, 5, 5, 3, 3, 0, 2] * 10)
    field_preds   = np.array([4, 1, 1, 4, 5, 3, 3, 2, 0, 3] * 10) # 60% correct

    # TODO 2: Calculate accuracy on Validation Set and Field Test Set
    # Hint: Accuracy is the fraction of correct predictions.
    # np.mean(preds == targets) returns this fraction.
    
    val_accuracy = 0.0 # TODO: Calculate validation accuracy
    field_accuracy = 0.0 # TODO: Calculate field accuracy

    print("=========================================")
    print("Out-of-Distribution (OOD) Evaluation:")
    print("=========================================")
    print(f"Validation (In-Distribution) Accuracy: {val_accuracy * 100:.2f}%")
    print(f"Field Test (Out-of-Distribution) Accuracy: {field_accuracy * 100:.2f}%")
    print(f"Accuracy Gap (Domain Shift): {(val_accuracy - field_accuracy) * 100:.2f}%\n")

    # TODO 3: Plot a comparative bar chart of the two accuracies
    # Requirements:
    #   - Customize bar colors
    #   - Add gridlines, title, axis labels
    #   - Set y-limit [0, 1.0]
    #   - Annotate values on top of the bars
    
    labels = ['Validation Set (ID)', 'Field Test Set (OOD)']
    accuracies = [val_accuracy, field_accuracy]

    plt.figure(figsize=(8, 5))
    
    # TODO: Write matplotlib comparative bar chart code here
    
    
    # Save the evaluation plot
    os.makedirs("outputs", exist_ok=True)
    output_img = "outputs/field_vs_val_comparison.png"
    plt.savefig(output_img, dpi=300)
    print(f"Successfully generated and saved plot to '{output_img}'.")
    plt.show()

if __name__ == "__main__":
    main()
