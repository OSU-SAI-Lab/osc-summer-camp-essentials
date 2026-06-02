# ============================================================
#  🌿 OSC STEM INSTITUTE — DAY 1 FILLER (PART 3 OF 3)
#  Topic  : Introduction to NumPy and PyTorch
#  Script : ANSWER SCRIPT  ← instructors / after exercises only!
#  Time   : Review session
#  Tools  : Google Colab or Kaggle Notebook
# ============================================================
#
#  ⚠️  STUDENTS: Try the EXERCISES script first!
#      Only open this file after you've made a genuine attempt.
#
#  Each answer includes:
#    → The working code
#    → A short explanation of WHY it works
#    → Common mistakes to watch out for
#
# ============================================================

import os
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import time


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
#  PART A — NumPy Basics (ANSWERS)
# ============================================================

print("=" * 55)
print("PART A ANSWERS: NumPy Basics")
print("=" * 55)


# ── A1 ANSWER 🟢 ─────────────────────────────────────────────
print("\n── A1: Create an array ──")

temperatures = np.array([22, 25, 19, 30, 28, 21, 26])
print("Array:", temperatures)
print("Data type:", temperatures.dtype)

# WHY: np.array() wraps a Python list into a NumPy array.
# dtype int64 means each value is a 64-bit integer.
# You can force float: np.array([...], dtype=np.float32)


# ── A2 ANSWER 🟢 ─────────────────────────────────────────────
print("\n── A2: Array statistics ──")

print("Highest temperature:", temperatures.max())
print("Lowest temperature :", temperatures.min())
print("Average temperature:", temperatures.mean().round(2))
print("Temps above 24     :", (temperatures > 24).sum())  # = 4 (25, 30, 28, 26)

# WHY: .max(), .min(), .mean() are built-in NumPy methods.
# (temperatures > 24) creates a boolean array [F, T, F, T, T, F, T].
# .sum() on a boolean array counts the True values (True = 1, False = 0).
# Result = 4 because 25, 30, 28, and 26 are all > 24.
# COMMON MISTAKE: len(temperatures > 24) returns 7 (total length),
#                 not the count of True values. Always use .sum()!


# ── A3 ANSWER 🟢 ─────────────────────────────────────────────
print("\n── A3: linspace ──")

evenly_spaced = np.linspace(0, 1, 10)
print("10 evenly spaced values from 0 to 1:")
print(evenly_spaced.round(4))

# WHY: np.linspace(start, stop, num) creates 'num' values
# INCLUDING both endpoints. np.arange() excludes the endpoint.
# COMMON MISTAKE: np.arange(0, 1, 0.1) gives 10 values but
#                 may have floating-point precision issues.
#                 Use linspace when you want an exact count.


# ── A4 ANSWER 🟡 ─────────────────────────────────────────────
print("\n── A4: Normalise pixel values ──")

pixels = np.array([0, 128, 255, 64, 192, 32])
normalised_pixels = pixels / 255.0
print("Original pixels  :", pixels)
print("Normalised (0–1) :", normalised_pixels.round(3))

# WHY: Dividing by 255 maps [0, 255] → [0.0, 1.0].
# Neural networks train much better on small, normalised values.
# If pixels were 0–255, weights would grow huge trying to match them.
# COMMON MISTAKE: pixels / 255 with integer array gives integer division
#                 in older Python. Use 255.0 to force float output.
# NOTE: In real pipelines we also subtract mean and divide by std.
#       That's the Normalize() transform we use in torchvision!


# ── A5 ANSWER 🟡 ─────────────────────────────────────────────
print("\n── A5: 2D array (image grid) ──")

image_grid = np.arange(1, 17).reshape(4, 4)
print("Full 4x4 array:\n", image_grid)
print("First row       :", image_grid[0])         # or image_grid[0, :]
print("Last column     :", image_grid[:, -1])
print("Top-left 2x2:\n", image_grid[:2, :2])

# WHY: NumPy indexing uses [row, column] notation.
# image_grid[0]    → entire row 0 (first row)
# image_grid[:, -1] → all rows (:), last column (-1)
# image_grid[:2, :2] → rows 0–1, columns 0–1
# COMMON MISTAKE: Forgetting the comma — image_grid[:2][:2] also
#                 works here but is less readable and slower for large arrays.


# ── A6 ANSWER 🟡 ─────────────────────────────────────────────
print("\n── A6: Boolean indexing ──")

leaf_brightness = np.array([210, 88, 175, 45, 230, 95, 160, 72])
dark_mask   = leaf_brightness < 100
dark_leaves = leaf_brightness[dark_mask]

print("Brightness values:", leaf_brightness)
print("Dark mask (<100) :", dark_mask)
print("Dark leaf values :", dark_leaves)
print("Number of dark leaves:", dark_mask.sum())

# WHY: Boolean indexing is one of NumPy's most powerful features.
# leaf_brightness < 100 creates [F, T, F, T, F, T, F, T]
# Using that as an index returns only the True positions.
# COMMON MISTAKE: Forgetting that the result is a NEW array —
#                 modifying dark_leaves does NOT change leaf_brightness.


# ── A7 ANSWER 🔴 ─────────────────────────────────────────────
print("\n── A7: CHALLENGE — Mini colour image ──")

np.random.seed(99)
colour_img = np.random.randint(0, 256, size=(3, 8, 8))

print("Shape:", colour_img.shape)
channel_means = colour_img.mean(axis=(1, 2))
print("Mean per channel (R, G, B):", channel_means.round(2))
print("Highest average channel   :", ["Red", "Green", "Blue"][channel_means.argmax()])

# WHY: shape (3, 8, 8) = 3 channels × 8 rows × 8 cols
# axis=(1, 2) means "average over height AND width" for each channel.
# .argmax() returns the index of the maximum value: 0=R, 1=G, 2=B.
# COMMON MISTAKE: Using axis=0 would average across channels (wrong).
# NOTE: This is the EXACT format PyTorch uses for image tensors!


# ============================================================
#  PART B — NumPy Math (ANSWERS)
# ============================================================

print("\n" + "=" * 55)
print("PART B ANSWERS: NumPy Math & Operations")
print("=" * 55)


# ── B1 ANSWER 🟢 ─────────────────────────────────────────────
print("\n── B1: Accuracy statistics ──")

accuracy_scores = np.array([0.82, 0.87, 0.79, 0.91, 0.85])
print("Best accuracy :", accuracy_scores.max())
print("Mean accuracy :", accuracy_scores.mean().round(4))
print("Runs above 85%:", (accuracy_scores > 0.85).sum())

# WHY: straightforward aggregation. Note 0.85 not 85 —
# the array stores fractions, not percentages.


# ── B2 ANSWER 🟢 ─────────────────────────────────────────────
print("\n── B2: Array arithmetic ──")

A = np.array([1, 2, 3])
B = np.array([4, 5, 6])
print("A + B   :", A + B)
print("A * B   :", A * B)
print("A ** 2  :", A ** 2)
print("sqrt(B) :", np.sqrt(B).round(4))

# WHY: All operations are element-wise (same position in each array).
# [1+4, 2+5, 3+6] = [5, 7, 9] for addition.
# COMMON MISTAKE: * is element-wise, NOT matrix multiplication.
#                 Use np.dot(A, B) or A @ B for dot/matrix product.


# ── B3 ANSWER 🟡 ─────────────────────────────────────────────
print("\n── B3: Speed comparison ──")

SIZE = 500_000
py_list  = list(range(SIZE))
np_array = np.arange(SIZE, dtype=np.float64)

start   = time.time()
py_res  = [x * 2.0 for x in py_list]
py_time = time.time() - start

start   = time.time()
np_res  = np_array * 2.0
np_time = time.time() - start

print(f"Python list : {py_time:.5f} sec")
print(f"NumPy array : {np_time:.5f} sec")
print(f"NumPy is ~{py_time / max(np_time, 1e-9):.0f}x faster")

# WHY: NumPy uses compiled C code under the hood.
# Python loops are interpreted one step at a time.
# NumPy sends the whole operation to optimised C in one call.


# ── B4 ANSWER 🔴 ─────────────────────────────────────────────
print("\n── B4: CHALLENGE — Matrix multiplication ──")

W = np.array([[1, 2], [3, 4], [5, 6]])
x = np.array([[7], [8]])

result = W @ x           # same as np.dot(W, x)
print("W (weights):\n", W)
print("x (input):\n", x)
print("W @ x (output):\n", result)
print("Result shape:", result.shape)

# WHY: (3, 2) @ (2, 1) = (3, 1).
# The inner dimensions must match: W has 2 columns, x has 2 rows ✅
# Manual check: row 0 = 1*7 + 2*8 = 23. Row 1 = 3*7 + 4*8 = 53. Row 2 = 5*7 + 6*8 = 83.
# COMMON MISTAKE: Confusing * (element-wise) with @ (matrix multiply).
#                 W * x would fail here because shapes (3,2) and (2,1) don't broadcast.


# ============================================================
#  PART C — PyTorch Tensors (ANSWERS)
# ============================================================

print("\n" + "=" * 55)
print("PART C ANSWERS: PyTorch Tensors")
print("=" * 55)


# ── C1 ANSWER 🟢 ─────────────────────────────────────────────
print("\n── C1: Create a tensor ──")

leaf_scores = torch.tensor([0.9, 0.2, 0.85, 0.4, 0.95])
print("Tensor:", leaf_scores)
print("Shape :", leaf_scores.shape)
print("dtype :", leaf_scores.dtype)

# WHY: torch.tensor() infers the dtype from the data.
# Floats → torch.float32. Integers → torch.int64.
# COMMON MISTAKE: torch.Tensor([...]) (capital T) is slightly different —
#                 always use torch.tensor() (lowercase t) for consistency.


# ── C2 ANSWER 🟢 ─────────────────────────────────────────────
print("\n── C2: Creating tensors ──")

zeros  = torch.zeros(3, 3)
ones   = torch.ones(2, 4)
torch.manual_seed(7)
random = torch.rand(3, 3)

print("zeros(3,3):\n", zeros)
print("ones(2,4):\n", ones)
print("rand(3,3):\n", random.round(decimals=3))

# WHY: torch.zeros and torch.ones are the most common initialisers.
# torch.rand gives uniform [0,1] values — good for initial exploration.
# torch.randn gives standard normal values — used to initialise weights.


# ── C3 ANSWER 🟢 ─────────────────────────────────────────────
print("\n── C3: NumPy ↔ PyTorch conversion ──")

np_data        = np.array([10.0, 20.0, 30.0, 40.0])
tensor_data    = torch.from_numpy(np_data)
half_tensor    = tensor_data * 0.5
back_to_numpy  = half_tensor.numpy()

print("Original NumPy  :", np_data)
print("As Tensor       :", tensor_data)
print("x 0.5 (tensor)  :", half_tensor)
print("Back to NumPy   :", back_to_numpy)

# WHY: torch.from_numpy() shares memory with the NumPy array —
#      modifying one changes the other! Use .clone() if you want a copy.
# COMMON MISTAKE: Calling .numpy() on a tensor that has requires_grad=True
#                 will raise an error. Use .detach().numpy() in that case.


# ── C4 ANSWER 🟡 ─────────────────────────────────────────────
print("\n── C4: GPU or CPU device ──")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

random_tensor = torch.rand(3, 3).to(device)
print("Tensor:\n", random_tensor.round(decimals=3))
print("Stored on:", random_tensor.device)

# WHY: .to(device) moves the tensor to GPU if available, else keeps on CPU.
# Training on GPU is ~10–100x faster because GPUs have thousands of cores
# that can compute tensor operations in parallel.
# COMMON MISTAKE: Mixing CPU and GPU tensors in one operation causes an error.
#                 Always move BOTH model AND data to the same device!


# ── C5 ANSWER 🟡 ─────────────────────────────────────────────
print("\n── C5: Image batch tensor shape ──")

# 8 colour images, 32x32 pixels → shape (batch, channels, H, W)
image_batch = torch.zeros(8, 3, 32, 32)
print("Batch shape:", image_batch.shape)  # torch.Size([8, 3, 32, 32])

# Flatten each image to a 1D vector
flat_batch = image_batch.view(8, -1)
print("Flat shape :", flat_batch.shape)   # torch.Size([8, 3072])
print("Numbers per image:", flat_batch.shape[1],
      f"  (= 3 × 32 × 32 = {3*32*32})")

# WHY: CNNs expect (batch, channels, height, width) — this is PyTorch's default.
# .view(8, -1) keeps 8 rows (one per image) and auto-calculates columns.
# 3 × 32 × 32 = 3072 numbers per image.
# COMMON MISTAKE: Using .reshape() vs .view() — .view() requires contiguous
#                 memory; when unsure, use .reshape() which handles both cases.


# ── C6 ANSWER 🟡 ─────────────────────────────────────────────
print("\n── C6: Tensor math ──")

a = torch.tensor([2.0, 4.0, 6.0, 8.0])
b = torch.tensor([1.0, 3.0, 5.0, 7.0])
print("a:", a)
print("b:", b)
print("a + b    :", a + b)
print("a - b    :", a - b)
print("a * b    :", a * b)
print("sqrt(a)  :", torch.sqrt(a))
print("mean(a)  :", a.mean().item())
print("max(b)   :", b.max().item())

# WHY: All operations identical to NumPy — intentional design.
# .item() converts a scalar tensor to a plain Python float.
# COMMON MISTAKE: Printing the tensor shows extra "tensor()" wrapper.
#                 Use .item() for scalars, .numpy() for arrays.


# ── C7 ANSWER 🔴 ─────────────────────────────────────────────
print("\n── C7: CHALLENGE — Autograd ──")

x = torch.tensor(4.0, requires_grad=True)
y = 3 * x**2 + 5 * x - 2
y.backward()

print("x =", x.item())
print("y = 3x² + 5x - 2 =", y.item())
print("dy/dx (computed by PyTorch):", x.grad.item())
print("dy/dx (by hand: 6x + 5 at x=4):", 6*4 + 5, "✅")

# WHY: PyTorch builds a computation graph as you run operations.
# .backward() walks back through this graph applying the chain rule.
# x.grad stores the accumulated gradient ∂y/∂x.
# This is the mathematical engine behind ALL neural network training!
# COMMON MISTAKE: Calling .backward() twice accumulates gradients.
#                 Always call optimizer.zero_grad() before each training step.


# ============================================================
#  PART D — Putting It Together (ANSWERS)
# ============================================================

print("\n" + "=" * 55)
print("PART D ANSWERS: Putting It Together")
print("=" * 55)


# ── D1 ANSWER 🟡 ─────────────────────────────────────────────
print("\n── D1: Softmax and predictions ──")

class_names = ["Healthy", "Early Blight", "Late Blight"]
logits      = torch.tensor([1.5, 3.2, 0.8])

probabilities  = torch.softmax(logits, dim=0)
predicted_idx  = probabilities.argmax().item()
predicted_class = class_names[predicted_idx]
confidence     = probabilities[predicted_idx].item()

print("Raw logits     :", logits.tolist())
print("Probabilities  :", [round(p, 3) for p in probabilities.tolist()])
print("Predicted index:", predicted_idx)
print("Predicted class:", predicted_class)
print(f"Confidence     : {confidence:.1%}")

# WHY: Softmax converts any numbers into probabilities summing to 1.
# High logit → high probability. The highest logit wins.
# .argmax() returns the position of the largest value.
# .item() converts a single-element tensor to a plain Python number.
# COMMON MISTAKE: dim=0 for a 1D tensor, dim=1 for a 2D batch.
#                 In real batched prediction, always use dim=1.


# ── D2 ANSWER 🟡 ─────────────────────────────────────────────
print("\n── D2: Plot a loss curve ──")

losses = [2.3, 1.9, 1.5, 1.2, 1.0, 0.85, 0.75, 0.68, 0.63, 0.60]
epochs = np.arange(1, 11)

plt.figure(figsize=(8, 4))
plt.plot(epochs, losses, "g-o", linewidth=2, markersize=7)
plt.title("My Training Loss Curve", fontsize=13)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("✅ Loss curve plotted!")

# WHY: We use matplotlib to visualise how training is going.
# A healthy loss curve starts high and decreases, eventually flattening.
# If it goes UP or stays flat, training is not working.


# ── D3 ANSWER 🟡 ─────────────────────────────────────────────
print("\n── D3: Read and explore a CSV with pandas ──")

df = pd.read_csv(CSV_PATH)

print("First 5 rows:")
print(df.head())
print("\nShape (rows, columns):", df.shape)
print("\nData type of each column:")
print(df.dtypes)
print(f"\nAverage {LENGTH_COLUMN}:", df[LENGTH_COLUMN].mean().round(3))

# WHY: pd.read_csv() loads a whole spreadsheet in one line.
# Using the GLOBAL CSV_PATH means this works no matter where the
# script is launched from — we never hard-code the path inline.
# .head() previews the top rows, .shape gives (rows, columns),
# and .dtypes shows that each column has its OWN type:
# int64 (sample_id), object/str (species), float64 (measurements), bool (is_healthy).
# COMMON MISTAKE: Hard-coding "sample_data/leaf_measurements.csv" as a
#                 relative path — it breaks the moment you run the file
#                 from a different folder. Build the path from BASE_DIR instead.


# ── D4 ANSWER 🔴 ─────────────────────────────────────────────
print("\n── D4: CHALLENGE — Filter, group, and convert to NumPy ──")

df = pd.read_csv(CSV_PATH)   # re-read so D4 stands on its own

# a) Count the healthy leaves with a boolean filter
healthy = df[df[HEALTH_COLUMN] == True]
print("Number of healthy leaves:", len(healthy))

# b) Average chlorophyll per species with groupby
species_means = df.groupby(SPECIES_COLUMN)[CHLOROPHYLL_COLUMN].mean()
print("\nAverage chlorophyll per species:")
print(species_means.round(3))

# c) Pull the column out as a NumPy array and take its mean
chlorophyll_np = df[CHLOROPHYLL_COLUMN].to_numpy()
pandas_mean = df[CHLOROPHYLL_COLUMN].mean()
numpy_mean  = chlorophyll_np.mean()

print("\npandas mean:", round(pandas_mean, 4))
print("NumPy mean :", round(numpy_mean, 4))

# d) Confirm they match
print("Means match:", np.isclose(pandas_mean, numpy_mean))

# WHY: Filtering a DataFrame with df[condition] is the SAME boolean-mask
# idea you used on NumPy arrays in A6 — just on a labelled table.
# .groupby(col) splits the rows by category, then .mean() summarises each group.
# .to_numpy() hands the raw values to NumPy/PyTorch for the math — that
# pandas → NumPy handoff is the everyday workflow on real datasets.
# COMMON MISTAKE: Combining conditions without parentheses, e.g.
#                 df[df.a > 1 & df.b < 2] — always wrap each: (df.a > 1) & (df.b < 2).


# ============================================================
#  COMMON MISTAKES SUMMARY
# ============================================================

print("\n" + "="*55)
print("📋 COMMON MISTAKES TO REMEMBER")
print("="*55)
print("""
  NumPy:
    ❌ (arr > 100).len()  → use .sum() to count True values
    ❌ arr1 * arr2        → element-wise, NOT matrix multiply
    ✅ arr1 @ arr2        → use @ or np.dot() for matrix multiply
    ❌ arr / 255          → integer division if arr is int!
    ✅ arr / 255.0        → forces float output

  PyTorch:
    ❌ torch.Tensor([1,2,3])  → use torch.tensor() (lowercase t)
    ❌ y.backward() twice     → gradients accumulate! zero them first
    ❌ tensor.numpy()         → fails if requires_grad=True
    ✅ tensor.detach().numpy()→ always safe

  pandas:
    ❌ "data/file.csv" inline → breaks from another folder; build from BASE_DIR
    ❌ df[df.a > 1 & df.b < 2]→ wrap each condition: (df.a > 1) & (df.b < 2)
    ❌ df.column_name typo    → use the GLOBAL column-name constants
    ✅ df[COL].to_numpy()     → hand the values to NumPy/PyTorch for math

  Both:
    ❌ Mixing CPU/GPU tensors → always put model AND data on same device
    ❌ Plotting a tensor      → convert to numpy first (.detach().numpy())
""")

print("\n🌟 Great work! You now understand the building blocks of all AI.")
print("   These patterns — loading data with pandas, crunching it with")
print("   NumPy, and gradients in PyTorch — appear in EVERY project, including ours!")
