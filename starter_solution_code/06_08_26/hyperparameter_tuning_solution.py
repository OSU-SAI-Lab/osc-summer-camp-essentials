"""
Monday, June 8 - Afternoon Session
Hyperparameter Sweep Simulation (Solution Code)

What this script is:
A reference solution for hyperparameter searches.

Goal of this script:
Run learning rate sweeps, log trial results to JSON, and plot bar charts comparing performance.

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
    if lr == 1e-3:
        return 0.915, 0.231
    elif lr == 1e-2:
        return 0.584, 1.432
    elif lr == 1e-4:
        return 0.812, 0.456
    else:
        return 0.700, 0.700

def main():
    # TODO 1 Solution: Define search space
    learning_rates = [1e-2, 1e-3, 1e-4]

    results = []

    print("Starting simulated hyperparameter sweep...")
    for lr in learning_rates:
        print(f"Running experiment with learning_rate={lr}...")
        
        # Run simulation
        acc, loss = train_simulated_model(lr)
        
        trial_data = {
            "learning_rate": lr,
            "validation_accuracy": acc,
            "validation_loss": loss
        }
        results.append(trial_data)
        print(f"  Result: Accuracy={acc:.4f}, Loss={loss:.4f}")

    # TODO 2 Solution: Save to JSON file
    os.makedirs("outputs", exist_ok=True)
    output_json = "outputs/tuning_results.json"
    with open(output_json, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\nAll trial results saved to '{output_json}'.")

    # Extract values for plotting
    lr_labels = [str(r["learning_rate"]) for r in results]
    accuracies = [r["validation_accuracy"] for r in results]

    # TODO 3 Solution: Matplotlib bar chart
    plt.figure(figsize=(8, 5))
    bars = plt.bar(lr_labels, accuracies, color=["crimson", "seagreen", "dodgerblue"], width=0.5)
    
    # Customize plot
    plt.title("Hyperparameter Optimization: Learning Rate Sweep", fontsize=14, fontweight="bold")
    plt.xlabel("Learning Rate Setting", fontsize=12)
    plt.ylabel("Validation Accuracy", fontsize=12)
    plt.ylim(0, 1.0)
    plt.grid(axis='y', linestyle=':', alpha=0.6)
    
    # Annotate bar heights
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, height + 0.02, f"{height:.3f}",
                 ha='center', va='bottom', fontsize=11, fontweight="bold")
                 
    # Save the chart
    output_img = "outputs/tuning_comparison.png"
    plt.savefig(output_img, dpi=300)
    print(f"Successfully generated and saved plot to '{output_img}'.")
    plt.show()

if __name__ == "__main__":
    main()
