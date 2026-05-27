"""
Monday, June 8 - Morning Session
Error Analysis and Metrics Report (Starter Code)

What this script is:
A metrics report generator and confusion matrix plotter.

Goal of this script:
Generate classification reports (Precision, Recall, F1-Scores) for each disease class, and plot a confusion matrix heatmap.

Why we are doing it (Student Context):
A simple accuracy score hides model mistakes. Precision and recall tell us exactly which crop classes (e.g. Insect Damage vs Healthy) the model confuses, guiding our troubleshooting and data improvements.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

def main():
    # Target class names
    classes = [
        'DicambaDamage', 
        'FrogEyeLeafSpot', 
        'GenericFeeding', 
        'InsectDamage', 
        'Soybeans', 
        'SuddenDeathSyndrome'
    ]

    # Mock ground truth and predicted labels representing Model v1 baseline results.
    # 0 = DicambaDamage, 1 = FrogEyeLeafSpot, 2 = GenericFeeding,
    # 3 = InsectDamage,  4 = Soybeans,        5 = SuddenDeathSyndrome
    y_true = np.array([4, 4, 4, 1, 1, 1, 5, 5, 5, 3, 3, 3, 0, 0, 2, 2, 4, 1, 5, 3] * 5)
    
    # Simulating a model that confuses GenericFeeding (2) and InsectDamage (3), 
    # as well as healthy Soybeans (4) and FrogEyeLeafSpot (1)
    y_pred = np.array([4, 4, 1, 1, 1, 4, 5, 5, 5, 3, 2, 3, 0, 0, 2, 3, 4, 1, 5, 2] * 5)

    # TODO 1: Generate and print the text-based classification report
    # Hint: Use classification_report(y_true, y_pred, target_names=classes)
    
    print("=========================================")
    print("Classification Metrics Report:")
    print("=========================================")
    report = "" # TODO: Generate report
    print(report)

    # TODO 2: Calculate the confusion matrix
    # Hint: Use confusion_matrix(y_true, y_pred)
    
    cm = None # TODO: Calculate confusion matrix
    print("Confusion Matrix calculated.")

    # TODO 3: Plot the confusion matrix as an annotated heatmap using Matplotlib
    # Requirements:
    #   - Use plt.imshow() to show the matrix values
    #   - Add class labels to x-axis and y-axis ticks
    #   - Loop through the matrix and add text annotations inside each cell
    #   - Add axis labels and title
    
    plt.figure(figsize=(10, 8))
    
    # TODO: Write matplotlib heatmap plotting code here
    
    
    # Save the heatmap plot
    os.makedirs("outputs", exist_ok=True)
    output_img = "outputs/confusion_matrix.png"
    plt.savefig(output_img, dpi=300)
    print(f"\nSuccessfully generated and saved plot to '{output_img}'.")
    plt.show()

if __name__ == "__main__":
    main()
