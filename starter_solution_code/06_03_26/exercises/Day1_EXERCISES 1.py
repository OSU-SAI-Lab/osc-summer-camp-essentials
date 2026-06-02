# ============================================================
#  🌿 OSC STEM INSTITUTE — DAY 1 FILLER (PART 2 OF 3)
#  Topic  : Introduction to NumPy and PyTorch
#  Script : STUDENT EXERCISES  ← you write the code here!
#  Time   : ~1.5 hours
#  Tools  : Google Colab or Kaggle Notebook
# ============================================================
#
#  INSTRUCTIONS:
#    → Read each question carefully
#    → Write your answer where it says  # YOUR CODE HERE
#    → Run the cell to check your answer
#    → DO NOT look at the answer script yet — try first!
#    → If you're stuck for more than 5 minutes, ask your instructor
#
#  DIFFICULTY LEVELS:
#    🟢 EASY      — you should be able to do this right away
#    🟡 MEDIUM    — needs a little thinking
#    🔴 CHALLENGE — for fast finishers!
#
# ============================================================

import os
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
#  GLOBAL CONFIG — set these in ONE place, use them everywhere
# ============================================================
# These globals point at the CSV in the "sample_data" folder next
# to this script, and name its columns. If the data ever moves or a
# column is renamed, you only change it HERE. (The try/except lets
# it work both as a .py file and inside a Colab/Kaggle notebook.)
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()

SAMPLE_DATA_DIR    = os.path.join(BASE_DIR, "sample_data")
CSV_FILENAME       = "leaf_measurements.csv"
CSV_PATH           = os.path.join(SAMPLE_DATA_DIR, CSV_FILENAME)

ID_COLUMN          = "sample_id"
SPECIES_COLUMN     = "species"
LENGTH_COLUMN      = "leaf_length_cm"
CHLOROPHYLL_COLUMN = "chlorophyll_index"
HEALTH_COLUMN      = "is_healthy"


# ============================================================
#  PART A — NumPy Basics
# ============================================================

print("=" * 50)
print("PART A: NumPy Basics")
print("=" * 50)


# ── A1 🟢 ────────────────────────────────────────────────────
# Create a NumPy array called 'temperatures' containing:
# [22, 25, 19, 30, 28, 21, 26]
# Then print the array and its data type.

print("\n── A1: Create an array ──")
# YOUR CODE HERE


# ── A2 🟢 ────────────────────────────────────────────────────
# Using the 'temperatures' array from A1:
# a) Print the highest temperature
# b) Print the lowest temperature
# c) Print the average (mean) temperature
# d) Print how many temperatures are above 24 degrees

print("\n── A2: Array statistics ──")
# YOUR CODE HERE


# ── A3 🟢 ────────────────────────────────────────────────────
# Create an array of 10 evenly spaced numbers between 0 and 1.
# (Hint: look up np.linspace)
# Print the array.

print("\n── A3: linspace ──")
# YOUR CODE HERE


# ── A4 🟡 ────────────────────────────────────────────────────
# A camera captures pixel brightness as a number from 0 to 255.
# We need to normalise these to be between 0 and 1
# (this is what we do before feeding images to a neural network!)
#
# Given:
pixels = np.array([0, 128, 255, 64, 192, 32])
#
# Divide every pixel value by 255 to get values between 0 and 1.
# Store the result in 'normalised_pixels' and print it.
# Round to 3 decimal places for display.

print("\n── A4: Normalise pixel values ──")
# YOUR CODE HERE


# ── A5 🟡 ────────────────────────────────────────────────────
# Create a 4x4 array (a tiny grayscale image!) using np.arange
# filled with values 1 to 16, reshaped to (4, 4).
#
# Then:
# a) Print the full array
# b) Print just the first row
# c) Print just the last column
# d) Print the top-left 2x2 corner

print("\n── A5: 2D array (image grid) ──")
# YOUR CODE HERE


# ── A6 🟡 ────────────────────────────────────────────────────
# You have brightness measurements from 8 leaves.
# Find which leaves are "dark" (brightness < 100) —
# these might be diseased!
#
# Given:
leaf_brightness = np.array([210, 88, 175, 45, 230, 95, 160, 72])
#
# a) Create a boolean mask for dark leaves (brightness < 100)
# b) Print the mask
# c) Print only the dark brightness values
# d) Print how many dark leaves there are

print("\n── A6: Boolean indexing ──")
# YOUR CODE HERE


# ── A7 🔴 ────────────────────────────────────────────────────
# CHALLENGE: A real image has 3 channels (R, G, B).
# Create a random (3, 8, 8) NumPy array representing a tiny
# 8x8 colour image. (Use np.random.randint with low=0, high=256)
#
# Then:
# a) Print the shape
# b) Print the mean value of each channel separately
#    Hint: axis=(1, 2) gives you the mean over height and width
# c) Which channel has the highest average value?

print("\n── A7: CHALLENGE — Mini colour image ──")
np.random.seed(99)
# YOUR CODE HERE


# ============================================================
#  PART B — NumPy Math & Operations
# ============================================================

print("\n" + "=" * 50)
print("PART B: NumPy Math & Operations")
print("=" * 50)


# ── B1 🟢 ────────────────────────────────────────────────────
# You have model accuracy scores from 5 training runs:
accuracy_scores = np.array([0.82, 0.87, 0.79, 0.91, 0.85])
#
# a) What is the best (maximum) accuracy?
# b) What is the average accuracy?
# c) How many runs scored above 85%?

print("\n── B1: Accuracy statistics ──")
# YOUR CODE HERE


# ── B2 🟢 ────────────────────────────────────────────────────
# Vector A = [1, 2, 3] and Vector B = [4, 5, 6]
# Compute:
# a) A + B
# b) A * B   (element-wise multiplication)
# c) A squared (every element to the power of 2)
# d) The square root of every element in B

print("\n── B2: Array arithmetic ──")
A = np.array([1, 2, 3])
B = np.array([4, 5, 6])
# YOUR CODE HERE


# ── B3 🟡 ────────────────────────────────────────────────────
# Compare speed: Python list vs NumPy
# Run a multiplication on 500,000 numbers each way and time them.
# Use the code structure from the teaching script (Section 2.8).
# Print both times and how much faster NumPy is.

import time
SIZE = 500_000
print("\n── B3: Speed comparison ──")
# YOUR CODE HERE


# ── B4 🔴 ────────────────────────────────────────────────────
# CHALLENGE: Matrix multiplication
# In neural networks, every layer does matrix multiplication.
# Given:
W = np.array([[1, 2], [3, 4], [5, 6]])   # shape (3, 2) — "weights"
x = np.array([[7], [8]])                  # shape (2, 1) — "input"
#
# Compute the matrix multiplication W @ x using np.dot() or @
# Print the result and its shape.
# (This is literally what a neural network layer does!)

print("\n── B4: CHALLENGE — Matrix multiplication ──")
# YOUR CODE HERE


# ============================================================
#  PART C — PyTorch Tensors
# ============================================================

print("\n" + "=" * 50)
print("PART C: PyTorch Tensors")
print("=" * 50)


# ── C1 🟢 ────────────────────────────────────────────────────
# Create a PyTorch tensor called 'leaf_scores' containing:
# [0.9, 0.2, 0.85, 0.4, 0.95]
# Print the tensor, its shape, and its data type (dtype).

print("\n── C1: Create a tensor ──")
# YOUR CODE HERE


# ── C2 🟢 ────────────────────────────────────────────────────
# Create the following tensors and print each one:
# a) A 3x3 tensor of all zeros
# b) A 2x4 tensor of all ones
# c) A 3x3 tensor of random values (use torch.rand, seed=7)

print("\n── C2: Creating tensors ──")
# YOUR CODE HERE


# ── C3 🟢 ────────────────────────────────────────────────────
# Convert between NumPy and PyTorch:
# a) Start with this NumPy array:
np_data = np.array([10.0, 20.0, 30.0, 40.0])
# b) Convert it to a PyTorch tensor
# c) Multiply every element by 0.5 using PyTorch operations
# d) Convert the result back to a NumPy array
# Print all three (original, tensor, final numpy)

print("\n── C3: NumPy ↔ PyTorch conversion ──")
# YOUR CODE HERE


# ── C4 🟡 ────────────────────────────────────────────────────
# Check which device is available (GPU or CPU).
# Create a tensor of shape (3, 3) filled with random values.
# Move it to the available device.
# Print the tensor and its device.

print("\n── C4: GPU or CPU device ──")
# YOUR CODE HERE


# ── C5 🟡 ────────────────────────────────────────────────────
# Tensor shapes — this is critical for understanding neural nets!
#
# In our CNN, we work with image batches shaped:
# (batch_size, channels, height, width)
#
# Create a tensor that represents:
# "A batch of 8 colour images, each 32x32 pixels"
# Use torch.zeros and print the shape.
#
# Then reshape it to a flat vector per image:
# Hint: use .view(8, -1)   — the -1 means "figure it out automatically"
# Print the new shape. How many numbers per image?

print("\n── C5: Image batch tensor shape ──")
# YOUR CODE HERE


# ── C6 🟡 ────────────────────────────────────────────────────
# Tensor math:
# a = [2.0, 4.0, 6.0, 8.0]   as a tensor
# b = [1.0, 3.0, 5.0, 7.0]   as a tensor
#
# Compute:
# a) a + b
# b) a - b
# c) a * b
# d) torch.sqrt(a)
# e) The mean of a
# f) The max value in b

print("\n── C6: Tensor math ──")
# YOUR CODE HERE


# ── C7 🔴 ────────────────────────────────────────────────────
# CHALLENGE: Autograd — PyTorch's learning superpower!
#
# Create a tensor x = 4.0 with requires_grad=True
# Compute: y = 3 * x**2 + 5 * x - 2
#
# Call y.backward() to compute the gradient.
# Print x.grad — the gradient of y with respect to x.
#
# BONUS: Calculate what the gradient should be by hand
# (Hint: dy/dx = 6x + 5, evaluate at x=4)
# Does your answer match?

print("\n── C7: CHALLENGE — Autograd ──")
# YOUR CODE HERE


# ============================================================
#  PART D — Putting It Together
# ============================================================

print("\n" + "=" * 50)
print("PART D: Putting It Together")
print("=" * 50)


# ── D1 🟡 ────────────────────────────────────────────────────
# Simulate what our model does when it makes a prediction.
#
# A model outputs RAW SCORES (called logits) for 3 classes:
# ["Healthy", "Early Blight", "Late Blight"]
#
logits = torch.tensor([1.5, 3.2, 0.8])
#
# Apply softmax to convert these to PROBABILITIES (they must sum to 1).
# Use torch.softmax(logits, dim=0)
#
# a) Print the probabilities (round to 3 decimal places)
# b) Print the predicted class INDEX (hint: use .argmax())
# c) Print the predicted CLASS NAME
# d) Print the confidence % for the top prediction

print("\n── D1: Softmax and predictions ──")
class_names = ["Healthy", "Early Blight", "Late Blight"]
# YOUR CODE HERE


# ── D2 🟡 ────────────────────────────────────────────────────
# Plot a simple learning curve using NumPy + Matplotlib.
#
# Simulate 10 epochs of training with these loss values:
losses = [2.3, 1.9, 1.5, 1.2, 1.0, 0.85, 0.75, 0.68, 0.63, 0.60]
#
# Create a line plot:
# - X axis: epoch number (1 to 10)
# - Y axis: loss value
# - Title: "My Training Loss Curve"
# - X label: "Epoch"
# - Y label: "Loss"
# - Use a green line with circle markers

print("\n── D2: Plot a loss curve ──")
# YOUR CODE HERE


# ── D3 🟡 ────────────────────────────────────────────────────
# Load and explore a REAL data table with pandas.
#
# A CSV of leaf measurements lives in the "sample_data" folder.
# Use the GLOBAL variable CSV_PATH (defined at the top) to read it.
#
# a) Read the file into a DataFrame called 'df' (hint: pd.read_csv)
# b) Print the first 5 rows (hint: .head())
# c) Print the shape — how many rows and columns?
# d) Print the data type of each column (hint: .dtypes)
# e) Print the average value of the LENGTH_COLUMN (use the global name)

print("\n── D3: Read and explore a CSV with pandas ──")
# YOUR CODE HERE


# ── D4 🔴 ────────────────────────────────────────────────────
# CHALLENGE: Filter, group, and hand data to NumPy.
#
# Using the same 'df' from D3 (re-read it if you need to):
#
# a) Count how many leaves are healthy
#    (filter rows where HEALTH_COLUMN == True, then use len())
# b) Use .groupby(SPECIES_COLUMN) to print the AVERAGE
#    CHLOROPHYLL_COLUMN value for each species
# c) Pull the CHLOROPHYLL_COLUMN out as a NumPy array
#    (hint: .to_numpy()) and print its mean using NumPy
# d) Confirm the pandas mean and the NumPy mean match!

print("\n── D4: CHALLENGE — Filter, group, and convert to NumPy ──")
# YOUR CODE HERE


# ============================================================
#  DONE! 🎉
# ============================================================
print("\n" + "="*50)
print("🎉 You finished the exercises!")
print("="*50)
print("""
  How did you do?
    A1–A5   → NumPy basics       ✅ or ❌?
    A6–A7   → Advanced indexing  ✅ or ❌?
    B1–B4   → NumPy math         ✅ or ❌?
    C1–C6   → PyTorch tensors    ✅ or ❌?
    C7      → Autograd           ✅ or ❌?
    D1–D2   → Predictions+plots  ✅ or ❌?
    D3–D4   → pandas data tables ✅ or ❌?

  Stuck on something? That's normal — it means you're learning!
  Check the answer script: Day1_Filler_ANSWERS.py
""")
