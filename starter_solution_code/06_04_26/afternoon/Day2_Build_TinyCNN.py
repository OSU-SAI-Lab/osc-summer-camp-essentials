# ============================================================
#  🌿 PLANT DISEASE DETECTION — DAY 2
#  Topic : How Does a Neural Network Learn? Build Your First CNN!
#  Time  : ~5 hours (with breaks)
#  Tools : Google Colab or Kaggle Notebook
# ============================================================
#
#  BY THE END OF TODAY YOU WILL:
#    ✅ Understand what a neural network is — visually
#    ✅ Know what a CNN (Convolutional Neural Network) does
#    ✅ Build a small CNN in PyTorch
#    ✅ Train it on 3 disease classes and see it learn!
#
# ============================================================


# ------------------------------------------------------------
# STEP 0 — Imports
# ------------------------------------------------------------

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import random

# Reproducibility — same results every run
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

print("✅ PyTorch version:", torch.__version__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("🖥️  Using device:", device)
# If you see 'cuda' — great, training will be fast!
# If you see 'cpu'  — still works, just a bit slower.


# ------------------------------------------------------------
# STEP 1 — What Is a Neural Network? (Read This!)
# ------------------------------------------------------------
#
#  Think of it like a chain of decisions:
#
#  Image → [Layer 1: spots edges] → [Layer 2: spots shapes]
#       → [Layer 3: spots patterns] → [Output: "Early Blight!"]
#
#  Each layer is made of NEURONS (tiny math functions).
#  Each neuron has a WEIGHT — how much it "pays attention" to its input.
#  Training = adjusting those weights until the answers are correct.
#
#  A CNN (Convolutional Neural Network) is special for IMAGES:
#    - It uses small filters (like a magnifying glass) that slide
#      across the image looking for patterns (edges, spots, textures)
#    - It shares these filters across the whole image (efficient!)
#    - Early layers: simple patterns (edges, colors)
#    - Later layers: complex patterns (leaf shapes, disease spots)
#
# ------------------------------------------------------------


# ------------------------------------------------------------
# STEP 2 — Pick 3 Classes to Work With Today
# ------------------------------------------------------------
# Training on all 38 classes takes a long time.
# Today we use just 3 classes so we can see results quickly.

DATASET_PATH = "../dataset/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)"
TRAIN_DIR = os.path.join(DATASET_PATH, "train")
VALID_DIR = os.path.join(DATASET_PATH, "valid")

# We'll pick 3 contrasting classes so the model has something clear to learn
SELECTED_CLASSES = [
    "Tomato___healthy",
    "Tomato___Early_blight",
    "Tomato___Late_blight"
]

print("📋 We're training on these 3 classes today:")
for i, cls in enumerate(SELECTED_CLASSES):
    print(f"   {i}: {cls}")


# ------------------------------------------------------------
# STEP 3 — Prepare the Data (Transforms & DataLoaders)
# ------------------------------------------------------------
# Before feeding images into a neural network, we must:
#   1. Resize all images to the same size (224x224 pixels)
#   2. Convert them to PyTorch tensors (arrays of numbers)
#   3. Normalize pixel values to a standard range
#      (helps the network learn more smoothly)

# Image size — we'll keep it small today for speed
IMG_SIZE   = 64    # 64x64 pixels (tiny but fast for learning)
BATCH_SIZE = 32    # how many images to feed the network at once

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),  # resize
    transforms.ToTensor(),                    # convert to tensor [0,1]
    transforms.Normalize(                     # normalize
        mean=[0.485, 0.456, 0.406],
        std =[0.229, 0.224, 0.225]
    )
])


# ------------------------------------------------------------
# STEP 3b — Filter Dataset to Our 3 Classes Only
# ------------------------------------------------------------
# ImageFolder loads ALL classes by default.
# We use a helper to keep only our 3 chosen classes.

from torch.utils.data import Subset

def get_filtered_dataset(root_dir, selected_classes, transform):
    """
    Load only the classes we want from a folder-based dataset.
    Returns a filtered Dataset and the class-to-index mapping.
    """
    full_dataset = ImageFolder(root=root_dir, transform=transform)

    # Find the index numbers for our chosen classes
    class_to_idx = full_dataset.class_to_idx
    selected_idx = {cls: class_to_idx[cls] for cls in selected_classes
                    if cls in class_to_idx}

    print(f"   Class → index mapping: {selected_idx}")

    # Keep only samples that belong to our 3 classes
    keep_indices = [i for i, (_, label) in enumerate(full_dataset.samples)
                    if label in selected_idx.values()]

    # Remap labels to 0, 1, 2 for simplicity
    idx_remap = {old: new for new, old in enumerate(selected_idx.values())}
    filtered_samples = [(path, idx_remap[label])
                        for path, label in full_dataset.samples
                        if label in selected_idx.values()]
    full_dataset.samples = filtered_samples
    full_dataset.targets = [label for _, label in filtered_samples]
    full_dataset.classes = selected_classes
    full_dataset.class_to_idx = {cls: i for i, cls in enumerate(selected_classes)}

    return Subset(full_dataset, range(len(filtered_samples)))


print("\n📂 Loading training data...")
train_dataset = get_filtered_dataset(TRAIN_DIR, SELECTED_CLASSES, transform)

print("\n📂 Loading validation data...")
valid_dataset = get_filtered_dataset(VALID_DIR, SELECTED_CLASSES, transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"\n✅ Training samples  : {len(train_dataset)}")
print(f"✅ Validation samples: {len(valid_dataset)}")
print(f"✅ Batches per epoch : {len(train_loader)}")


# ------------------------------------------------------------
# STEP 4 — Visualize a Batch of Training Images
# ------------------------------------------------------------
# Always look at your data before training!

def show_batch(loader, classes, num_images=8):
    images, labels = next(iter(loader))

    # Un-normalize for display
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    images_display = images * std + mean
    images_display = images_display.clamp(0, 1)

    fig, axes = plt.subplots(1, num_images, figsize=(16, 3))
    fig.suptitle("A batch of training images", fontsize=12)
    for i in range(num_images):
        ax = axes[i]
        img = images_display[i].permute(1, 2, 0).numpy()
        ax.imshow(img)
        ax.set_title(classes[labels[i].item()].split("__")[-1], fontsize=7)
        ax.axis("off")
    plt.tight_layout()
    plt.show()

show_batch(train_loader, SELECTED_CLASSES)


# ------------------------------------------------------------
# STEP 5 — Build Our CNN Model
# ------------------------------------------------------------
# This is the brain of our AI! Let's build it layer by layer.

class TinyCNN(nn.Module):
    """
    A small CNN for learning — simple enough to understand,
    powerful enough to detect plant diseases!

    Architecture:
      Input Image (3 x 64 x 64)
           ↓
      Conv Block 1: 32 filters  → spot simple patterns (edges)
           ↓
      Conv Block 2: 64 filters  → spot complex patterns (spots, textures)
           ↓
      Flatten → Fully Connected → Output (3 class scores)
    """

    def __init__(self, num_classes=3):
        super(TinyCNN, self).__init__()

        # --- FEATURE EXTRACTOR ---
        # Conv2d(in_channels, out_channels, kernel_size)
        # Think of kernel_size=3 as a 3x3 magnifying glass sliding over the image

        self.features = nn.Sequential(

            # Block 1 — find simple edges and color blobs
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # 3 input channels (RGB), 32 filters
            nn.ReLU(),                                    # ReLU: negative values → 0 (adds non-linearity)
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                           # shrink image: 64x64 → 32x32
            nn.Dropout(0.25),                             # randomly turn off 25% neurons (prevents memorizing)

            # Block 2 — find more complex patterns
            nn.Conv2d(32, 64, kernel_size=3, padding=1), # 32 → 64 filters
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                           # shrink image: 32x32 → 16x16
            nn.Dropout(0.25),
        )

        # --- CLASSIFIER ---
        # After feature extraction, flatten the image into a long list of numbers
        # then use fully connected layers to make the final decision
        self.classifier = nn.Sequential(
            nn.Flatten(),                  # 64 x 16 x 16 = 16,384 numbers
            nn.Linear(64 * 16 * 16, 256), # compress to 256 values
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)   # final output: one score per class
        )

    def forward(self, x):
        """
        This is the path one image takes through the network.
        x : a batch of images, shape = (batch_size, 3, 64, 64)
        """
        x = self.features(x)      # extract patterns
        x = self.classifier(x)    # decide the class
        return x                  # raw scores (we apply softmax later)


# Create the model and send it to our device (GPU or CPU)
model = TinyCNN(num_classes=len(SELECTED_CLASSES)).to(device)

# Print a summary of the model
print("🧠 Model architecture:\n")
print(model)

# Count total trainable parameters
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\n🔢 Total trainable parameters: {total_params:,}")
print("   (Each parameter is a weight the network will learn!)")


# ------------------------------------------------------------
# STEP 6 — Set Up the Training Ingredients
# ------------------------------------------------------------
#
#  Training needs 3 things:
#
#  1. LOSS FUNCTION — measures how wrong the model is
#     CrossEntropyLoss is standard for classification tasks
#     A score of 0 = perfect, higher = worse
#
#  2. OPTIMIZER — decides HOW to adjust weights to reduce loss
#     Adam is a popular optimizer that works well out of the box
#
#  3. LEARNING RATE — how big a step to take when adjusting weights
#     Too big  → model overshoots, bounces around, won't converge
#     Too small → model learns but very slowly
#     0.001 is a safe starting point

criterion = nn.CrossEntropyLoss()          # loss function
optimizer = optim.Adam(model.parameters(), lr=0.001)  # optimizer

print("✅ Loss function : CrossEntropyLoss")
print("✅ Optimizer     : Adam (lr=0.001)")


# ------------------------------------------------------------
# STEP 7 — The Training Loop
# ------------------------------------------------------------
# This is where the magic happens!
# We loop through all our training images many times (EPOCHS).
# Each time, the model makes a prediction, we measure the error,
# and we nudge the weights to do better next time.

def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Train the model for one full pass through the training data.
    Returns average loss and accuracy for this epoch.
    """
    model.train()  # tell PyTorch we're in training mode
    total_loss     = 0
    correct        = 0
    total          = 0

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)

        # --- Forward pass ---
        outputs = model(images)           # model makes predictions
        loss    = criterion(outputs, labels)  # measure the error

        # --- Backward pass (learning!) ---
        optimizer.zero_grad()  # clear old gradients
        loss.backward()        # calculate new gradients
        optimizer.step()       # update weights

        # --- Track stats ---
        total_loss += loss.item()
        predicted   = outputs.argmax(dim=1)  # pick the class with highest score
        correct    += (predicted == labels).sum().item()
        total      += labels.size(0)

        # Print progress every 10 batches
        if (batch_idx + 1) % 10 == 0:
            print(f"   Batch {batch_idx+1}/{len(loader)} "
                  f"| Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy


def evaluate(model, loader, criterion, device):
    """
    Evaluate the model on validation data (no weight updates here).
    Returns average loss and accuracy.
    """
    model.eval()   # tell PyTorch we're in evaluation mode
    total_loss = 0
    correct    = 0
    total      = 0

    with torch.no_grad():  # don't track gradients (saves memory)
        for images, labels in loader:
            images  = images.to(device)
            labels  = labels.to(device)
            outputs = model(images)
            loss    = criterion(outputs, labels)

            total_loss += loss.item()
            predicted   = outputs.argmax(dim=1)
            correct    += (predicted == labels).sum().item()
            total      += labels.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy


# ------------------------------------------------------------
# STEP 8 — Run Training!
# ------------------------------------------------------------
# We'll train for 5 epochs to keep it quick today.
# Watch how the accuracy improves with each epoch!

NUM_EPOCHS = 5

train_losses, train_accs = [], []
val_losses,   val_accs   = [], []

print(f"\n🚀 Starting training for {NUM_EPOCHS} epochs...\n")
print("=" * 60)

for epoch in range(1, NUM_EPOCHS + 1):
    print(f"\n📅 Epoch {epoch}/{NUM_EPOCHS}")
    print("-" * 40)

    train_loss, train_acc = train_one_epoch(
        model, train_loader, criterion, optimizer, device)

    val_loss, val_acc = evaluate(
        model, valid_loader, criterion, device)

    train_losses.append(train_loss)
    train_accs.append(train_acc)
    val_losses.append(val_loss)
    val_accs.append(val_acc)

    print(f"\n   📈 Train Loss: {train_loss:.4f}  |  Train Acc: {train_acc:.1f}%")
    print(f"   📉 Val   Loss: {val_loss:.4f}  |  Val   Acc: {val_acc:.1f}%")
    print("=" * 60)

print("\n✅ Training complete!")


# ------------------------------------------------------------
# STEP 9 — Plot the Learning Curves
# ------------------------------------------------------------
# A learning curve shows how well the model improved over time.
# We want to see LOSS going DOWN and ACCURACY going UP.

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
epochs = range(1, NUM_EPOCHS + 1)

# Loss
axes[0].plot(epochs, train_losses, "b-o", label="Training Loss")
axes[0].plot(epochs, val_losses,   "r-o", label="Validation Loss")
axes[0].set_title("Loss over Epochs\n(lower is better)", fontsize=12)
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Accuracy
axes[1].plot(epochs, train_accs, "b-o", label="Training Accuracy")
axes[1].plot(epochs, val_accs,   "r-o", label="Validation Accuracy")
axes[1].set_title("Accuracy over Epochs\n(higher is better)", fontsize=12)
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy (%)")
axes[1].set_ylim(0, 105)
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.suptitle("🌿 TinyCNN Learning Curves — Day 2", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("day2_learning_curves.png", dpi=150)
plt.show()
print("✅ Learning curves saved as day2_learning_curves.png")


# ------------------------------------------------------------
# STEP 10 — Save the Model
# ------------------------------------------------------------
# We save the model so we can use it again tomorrow!

torch.save(model.state_dict(), "tiny_cnn_day2.pth")
print("💾 Model saved as tiny_cnn_day2.pth")


# ------------------------------------------------------------
# 🎯 DAY 2 CHALLENGE — Think & Try!
# ------------------------------------------------------------
# 1. What does the accuracy look like after just 1 epoch?
#    After 5? Write down the numbers.
#
# 2. Look at the two learning curves. Which metric improved more?
#
# 3. EXTRA: Change NUM_EPOCHS to 10. Does accuracy keep going up?
#    At what point does it stop improving much?
#
# 4. DISCUSS: What do you think "overfitting" means?
#    Look at your train vs val accuracy curves for a clue!
#
# Your notes:
# → Epoch 1 accuracy :
# → Epoch 5 accuracy :
# → What I noticed   :
# ------------------------------------------------------------

print("\n🌟 Day 2 complete!")
print("   You just built and trained your first neural network. That's real AI! 🎉")
print("   Tomorrow: Transfer Learning — borrow a smarter brain to get even better results.")
