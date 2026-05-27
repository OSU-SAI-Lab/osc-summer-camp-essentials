"""
Friday, June 5 - Afternoon Session
Training Log Monitor and Plotter (Solution Code)

What this script is:
A reference solution for log parsing and curve plotting.

Goal of this script:
Parse metrics from training logs using regular expressions and generate curve plots using matplotlib.

Why we are doing it (Student Context):
Training a model is an experiment. We monitor loss curves to see if the model is learning (loss going down) or diverging, helping us optimize hyperparameters like learning rate.
"""

import os
import re
import glob
import sys
import matplotlib.pyplot as plt

# A sample mock training log string.
# If no log file is passed, the script will use this string as a fallback.
MOCK_LOG_DATA = """
Found 1200 images across 6 classes.
Classes: ['DicambaDamage', 'FrogEyeLeafSpot', 'GenericFeeding', 'InsectDamage', 'Soybeans', 'SuddenDeathSyndrome']

Training classifier head...

Epoch 1/5, Loss: 1.4520
Epoch 2/5, Loss: 0.9840
Epoch 3/5, Loss: 0.7610
Epoch 4/5, Loss: 0.6120
Epoch 5/5, Loss: 0.5180

Fine-tuning last transformer block...

Fine Epoch 1/3, Loss: 0.4320
Fine Epoch 2/3, Loss: 0.3140
Fine Epoch 3/3, Loss: 0.2050
Fine-tuning complete!
"""

def parse_logs(log_content):
    """
    Parses epoch numbers and losses from log file text content.
    Returns:
        head_epochs (list of int): list of head training epoch indices
        head_losses (list of float): list of head training losses
        fine_epochs (list of int): list of fine-tuning epoch indices
        fine_losses (list of float): list of fine-tuning losses
    """
    head_epochs, head_losses = [], []
    fine_epochs, fine_losses = [], []

    # TODO 1 Solution: Parse lines using regex
    for line in log_content.split('\n'):
        # Match head training epochs
        head_match = re.search(r"Epoch (\d+)/\d+,\s+Loss:\s+([\d\.]+)", line)
        if head_match:
            epoch_num = int(head_match.group(1))
            loss_val = float(head_match.group(2))
            head_epochs.append(epoch_num)
            head_losses.append(loss_val)
            
        # Match fine-tuning epochs
        fine_match = re.search(r"Fine Epoch (\d+)/\d+,\s+Loss:\s+([\d\.]+)", line)
        if fine_match:
            epoch_num = int(fine_match.group(1))
            loss_val = float(fine_match.group(2))
            fine_epochs.append(epoch_num)
            fine_losses.append(loss_val)

    return head_epochs, head_losses, fine_epochs, fine_losses

def main():
    # 1. Check if a log file was passed as a command-line argument
    if len(sys.argv) > 1:
        log_path = sys.argv[1]
    else:
        # 2. Try to auto-detect the newest slurm log file in the directory
        slurm_files = glob.glob("slurm-*.out") + glob.glob("slurm-training-*.out")
        if slurm_files:
            # Sort by modification time, newest first
            slurm_files.sort(key=os.path.getmtime, reverse=True)
            log_path = slurm_files[0]
            print(f"Auto-detected newest SLURM log file: '{log_path}'")
        else:
            log_path = "slurm-training.log"
    
    if os.path.exists(log_path):
        print(f"Reading logs from file: '{log_path}'...")
        with open(log_path, 'r') as f:
            content = f.read()
    else:
        print(f"Log file '{log_path}' not found. Falling back to mock logs...")
        content = MOCK_LOG_DATA
        
    h_eps, h_losses, f_eps, f_losses = parse_logs(content)
    
    print("\nParsed Results:")
    print(f"  Head Epochs: {h_eps}, Losses: {h_losses}")
    print(f"  Fine Epochs: {f_eps}, Losses: {f_losses}\n")
    
    if not h_losses:
        print("No metrics parsed. Check your parser regex or string match logic.")
        return

    # Combine epochs sequentially
    total_epochs = h_eps + [e + len(h_eps) for e in f_eps]
    total_losses = h_losses + f_losses
    
    # TODO 2 Solution: Matplotlib plotting
    plt.figure(figsize=(10, 5))
    
    # Plot head training
    plt.plot(h_eps, h_losses, label="Head Training (lr=1e-3)", color="darkorange", marker="o", linewidth=2)
    
    # Plot fine tuning
    fine_timeline = [e + len(h_eps) for e in f_eps]
    plt.plot(fine_timeline, f_losses, label="Fine-Tuning (lr=1e-5)", color="dodgerblue", marker="s", linewidth=2)
    
    # Draw separator line
    plt.axvline(x=len(h_eps) + 0.5, color="red", linestyle="--", alpha=0.7, label="Unfreeze Backbone Block")
    
    # Details
    plt.title("DINOv2 Soybean Disease Classifier Training Loss Curve", fontsize=14, fontweight="bold")
    plt.xlabel("Total Epoch Index", fontsize=12)
    plt.ylabel("Cross Entropy Loss", fontsize=12)
    plt.xticks(list(range(1, len(total_epochs) + 1)))
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper right", fontsize=11)
    
    # Save the plotted curve
    os.makedirs("outputs", exist_ok=True)
    output_img = "outputs/training_curve.png"
    plt.savefig(output_img, dpi=300)
    print(f"Successfully generated and saved plot to '{output_img}'.")
    plt.show()

if __name__ == "__main__":
    main()
