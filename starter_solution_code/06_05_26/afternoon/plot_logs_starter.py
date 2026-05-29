"""
Friday, June 5 - Afternoon Session
Training Log Monitor and Plotter (Starter Code)

What this script is:
A log parser and training curve plotter.

Goal of this script:
Find the newest SLURM training output, parse epoch losses using regular expressions, and plot a sequential training curve.

Why we are doing it (Student Context):
Training a model is an experiment. We monitor loss curves to see if the model is learning (loss going down) or diverging, helping us optimize hyperparameters like learning rate.
"""

import os
import re
import glob
import sys
import matplotlib.pyplot as plt



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

    # TODO 1: Parse the loss values for head training and fine-tuning.
    # Hint: Loop through each line in log_content.split('\n').
    # Look for lines containing "Epoch X/5, Loss: Y" or "Fine Epoch X/3, Loss: Y".
    # Use regular expressions (re.search) or simple string splits.
    
    for line in log_content.split('\n'):
        # Match head training epochs
        head_match = re.search(r"Epoch (\d+)/\d+,\s+Loss:\s+([\d\.]+)", line)
        if head_match:
            epoch_num = int(head_match.group(1))
            loss_val = float(head_match.group(2))
            # TODO: Append values to head lists
            
        # Match fine-tuning epochs
        fine_match = re.search(r"Fine Epoch (\d+)/\d+,\s+Loss:\s+([\d\.]+)", line)
        if fine_match:
            epoch_num = int(fine_match.group(1))
            loss_val = float(fine_match.group(2))
            # TODO: Append values to fine lists
            

    return head_epochs, head_losses, fine_epochs, fine_losses

def main():
    # Check if a log file was passed as a command-line argument, or try to auto-detect the newest slurm log
    if len(sys.argv) > 1:
        log_path = sys.argv[1]
    else:
        slurm_files = glob.glob("slurm-*.out") + glob.glob("slurm-training-*.out")
        if slurm_files:
            slurm_files.sort(key=os.path.getmtime, reverse=True)
            log_path = slurm_files[0]
            print(f"Auto-detected newest SLURM log file: '{log_path}'")
        else:
            log_path = "slurm-training.log"
            
    # Read log content
    if os.path.exists(log_path):
        print(f"Reading logs from file: '{log_path}'...")
        with open(log_path, 'r') as f:
            content = f.read()
    else:
        print(f"Error: Log file '{log_path}' not found.")
        print("Did you submit your training job via SLURM yet?")
        sys.exit(1)
        
    # Parse metrics
    h_eps, h_losses, f_eps, f_losses = parse_logs(content)
    
    print("\nParsed Results:")
    print(f"  Head Epochs: {h_eps}, Losses: {h_losses}")
    print(f"  Fine Epochs: {f_eps}, Losses: {f_losses}\n")
    
    if not h_losses:
        print("No metrics parsed. Check your parser regex or string match logic.")
        return

    # Combine epochs sequentially for a single continuous timeline
    # Head training is epochs 1 to 5, fine-tuning is epochs 6 to 8
    total_epochs = h_eps + [e + len(h_eps) for e in f_eps]
    total_losses = h_losses + f_losses
    
    # TODO 2: Create a line plot showing loss vs. epoch index
    # Customize the plot with gridlines, title, labels, and marker points.
    # Draw a vertical dashed line separating "Head Training" and "Fine-tuning".
    
    plt.figure(figsize=(10, 5))
    
    # TODO: Write your matplotlib plotting code here
    
    
    # Save the plotted curve
    os.makedirs("outputs", exist_ok=True)
    output_img = "outputs/training_curve.png"
    plt.savefig(output_img, dpi=300)
    print(f"Successfully generated and saved plot to '{output_img}'.")
    plt.show()

if __name__ == "__main__":
    main()
