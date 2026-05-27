"""
Monday, June 8 - Morning Session
Error Analysis and Metrics Report (Solution Code)

What this script is:
A reference solution for classification metrics and confusion matrix plotting.

Goal of this script:
Compute metrics reports and plot confusion matrices as annotated heatmaps.

Why we are doing it (Student Context):
A simple accuracy score hides model mistakes. Precision and recall tell us exactly which crop classes (e.g. Insect Damage vs Healthy) the model confuses, guiding our troubleshooting and data improvements.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

def main():
    classes = [
        'DicambaDamage', 
        'FrogEyeLeafSpot', 
        'GenericFeeding', 
        'InsectDamage', 
        'Soybeans', 
        'SuddenDeathSyndrome'
    ]

    y_true = np.array([4, 4, 4, 1, 1, 1, 5, 5, 5, 3, 3, 3, 0, 0, 2, 2, 4, 1, 5, 3] * 5)
    y_pred = np.array([4, 4, 1, 1, 1, 4, 5, 5, 5, 3, 2, 3, 0, 0, 2, 3, 4, 1, 5, 2] * 5)

    # TODO 1 Solution: Generate classification report
    print("=========================================")
    print("Classification Metrics Report:")
    print("=========================================")
    report = classification_report(y_true, y_pred, target_names=classes)
    print(report)

    # TODO 2 Solution: Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix calculated.")

    # TODO 3 Solution: Heatmap plotting
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Greens)
    plt.title("Soybean Disease Classifier Confusion Matrix", fontsize=14, fontweight="bold")
    plt.colorbar()
    
    # Tick marks
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, ha="right")
    plt.yticks(tick_marks, classes)

    # Text annotations in each cell
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black",
                     fontsize=12,
                     fontweight="bold")

    plt.tight_layout()
    plt.ylabel('True Class Label', fontsize=12)
    plt.xlabel('Predicted Class Label', fontsize=12)
    
    # Save the heatmap plot
    os.makedirs("outputs", exist_ok=True)
    output_img = "outputs/confusion_matrix.png"
    plt.savefig(output_img, dpi=300)
    print(f"\nSuccessfully generated and saved plot to '{output_img}'.")
    plt.show()

if __name__ == "__main__":
    main()
