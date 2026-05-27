"""
Monday, June 8 - Afternoon Session
Hyperparameter Sweep Simulation (Starter Code)

What this script is:
A hyperparameter search and evaluation script.

Goal of this script:
Simulate training with different learning rates, save sweep results to JSON, and plot comparison charts.

Why we are doing it (Student Context):
Hyperparameters (like learning rate) determine how models learn. We conduct systematic sweeps to find the optimal settings rather than guessing, ensuring our model converges reliably.
"""

import json
import os
import matplotlib.pyplot as plt

def train_simulated_model(lr):
    """
    Simulates training a model with a given learning rate.
    Returns:
        accuracy (float): Simulated model accuracy
        loss (float): Simulated model loss
    """
    # Simple simulation logic: learning rate 1e-3 is optimal,
    # 1e-2 is too high (diverges), 1e-4 is too slow
    if lr == 1e-3:
        return 0.915, 0.231
    elif lr == 1e-2:
        return 0.584, 1.432
    elif lr == 1e-4:
        return 0.812, 0.456
    else:
        return 0.700, 0.700

def main():
    # TODO 1: Define a search space list of learning rates to test
    # Values to include: 0.01 (1e-2), 0.001 (1e-3), and 0.0001 (1e-4)
    
    learning_rates = [] # TODO: Define learning rates

    results = []

    print("Starting simulated hyperparameter sweep...")
    for lr in learning_rates:
        print(f"Running experiment with learning_rate={lr}...")
        
        # Run simulation
        acc, loss = train_simulated_model(lr)
        
        # Record trial results
        trial_data = {
            "learning_rate": lr,
            "validation_accuracy": acc,
            "validation_loss": loss
        }
        results.append(trial_data)
        print(f"  Result: Accuracy={acc:.4f}, Loss={loss:.4f}")

    # TODO 2: Save the sweep results list to a JSON file named 'tuning_results.json'
    # Hint: Use json.dump(results, f, indent=4)
    
    os.makedirs("outputs", exist_ok=True)
    output_json = "outputs/tuning_results.json"
    # Write JSON save code here:
    
    print(f"\nAll trial results saved to '{output_json}'.")

    # Extract values for plotting
    lr_labels = [str(r["learning_rate"]) for r in results]
    accuracies = [r["validation_accuracy"] for r in results]

    # TODO 3: Plot a bar chart comparing validation accuracy across learning rates
    # Requirements:
    #   - Set bar labels as string values of learning rates
    #   - Customize bar colors, add grid, y-axis limit [0, 1]
    #   - Add title and axis labels
    
    plt.figure(figsize=(8, 5))
    
    # TODO: Write matplotlib bar chart plotting code here
    
    
    # Save the chart
    output_img = "outputs/tuning_comparison.png"
    plt.savefig(output_img, dpi=300)
    print(f"Successfully generated and saved plot to '{output_img}'.")
    plt.show()

if __name__ == "__main__":
    main()
