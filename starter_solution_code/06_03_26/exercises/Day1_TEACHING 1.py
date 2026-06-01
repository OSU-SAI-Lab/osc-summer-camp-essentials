# ============================================================
#  🌿 OSC STEM INSTITUTE — DAY 1 FILLER (PART 1 OF 3)
#  Topic  : Introduction to NumPy and PyTorch
#  Script : TEACHING SCRIPT  ← instructor runs this live
#  Time   : ~2 hours
#  Tools  : Google Colab or Kaggle Notebook
# ============================================================
#
#  HOW TO USE THIS SCRIPT:
#    → Run each cell one at a time
#    → Read the explanation ABOVE each cell before running
#    → Ask students to PREDICT the output before you reveal it
#    → Pause at every "💬 DISCUSS" comment for class discussion
#
# ============================================================


# ============================================================
#  SECTION 1 — WHY DO WE NEED SPECIAL TOOLS?
#  (Before NumPy and PyTorch — what did Python have?)
# ============================================================

# ----------------------------------------------------------
# 1.1 — The "naive" way: plain Python lists
# ----------------------------------------------------------
# Imagine we have 5 leaf brightness measurements.
# Let's try to do math on them using just Python lists.

brightness = [120, 85, 200, 67, 190]

# Task: multiply every value by 2 (simulating a brightness boost)
# With a plain list, we have to loop manually:

doubled = []
for val in brightness:
    doubled.append(val * 2)

print("Plain Python list — doubled brightness:")
print(doubled)

# That works... but imagine doing this for 150,000 pixels in a photo.
# And then doing it for 54,000 photos in our dataset.
# That would be VERY slow and a lot of code!


# ----------------------------------------------------------
# 1.2 — The same thing with NumPy (just one line!)
# ----------------------------------------------------------
import numpy as np

brightness_np = np.array([120, 85, 200, 67, 190])
doubled_np    = brightness_np * 2

print("\nNumPy array — doubled brightness:")
print(doubled_np)

# 💬 DISCUSS: "What changed? Why does this matter for AI?"
# Key point: NumPy operates on the WHOLE array at once.
# No loop needed. Way faster. Way less code.


# ============================================================
#  SECTION 2 — NUMPY DEEP DIVE
#  "The foundation of almost all data science in Python"
# ============================================================

print("\n" + "="*55)
print("SECTION 2: NumPy — Arrays, Math, and Speed")
print("="*55)

# ----------------------------------------------------------
# 2.1 — Creating arrays (different ways)
# ----------------------------------------------------------

# From a list
a = np.array([1, 2, 3, 4, 5])
print("\n1D array from list:", a)
print("   Type:", type(a))
print("   dtype (data type inside):", a.dtype)

# Range of numbers (like Python's range, but as an array)
b = np.arange(0, 10, 2)          # start, stop, step
print("\narange(0, 10, 2):", b)

# Evenly spaced numbers between two values
c = np.linspace(0, 1, 6)         # 6 numbers from 0 to 1
print("linspace(0, 1, 6):", c)

# All zeros — very common to initialise arrays
zeros = np.zeros(5)
print("zeros(5):", zeros)

# All ones
ones = np.ones(5)
print("ones(5):", ones)

# Random numbers between 0 and 1 (used a lot in AI!)
np.random.seed(42)               # seed = same result every time
rand = np.random.rand(5)
print("random.rand(5):", rand.round(3))


# ----------------------------------------------------------
# 2.2 — 2D Arrays (matrices) — this is how images work!
# ----------------------------------------------------------
print("\n--- 2D Arrays (Matrices) ---")

# A 3x3 matrix — like a tiny 3-pixel by 3-pixel grayscale image
pixel_grid = np.array([
    [100, 150, 200],
    [ 80, 120, 160],
    [ 50,  90, 130]
])

print("\nA 3x3 pixel grid:")
print(pixel_grid)
print("Shape:", pixel_grid.shape)    # (rows, columns)
print("Total elements:", pixel_grid.size)
print("Number of dimensions:", pixel_grid.ndim)

# 💬 DISCUSS: "A real 224x224 colour image has what shape?"
# Answer: (224, 224, 3) — height x width x RGB channels


# ----------------------------------------------------------
# 2.3 — Reshaping — very common in AI pipelines
# ----------------------------------------------------------
print("\n--- Reshaping Arrays ---")

flat = np.arange(12)             # [0, 1, 2, ..., 11]
print("Flat (1D):", flat)

grid = flat.reshape(3, 4)        # 3 rows, 4 columns
print("\nReshaped to (3, 4):")
print(grid)

back_to_flat = grid.flatten()
print("\nFlattened back:", back_to_flat)

# 💬 DISCUSS: "In our CNN on Day 2, we 'flatten' the feature map
#              before the final layer. Now you know what that means!"


# ----------------------------------------------------------
# 2.4 — Array Math — operates element-by-element
# ----------------------------------------------------------
print("\n--- Array Math ---")

x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
y = np.array([10.0, 20.0, 30.0, 40.0, 50.0])

print("x       :", x)
print("y       :", y)
print("x + y   :", x + y)           # element-wise addition
print("x * y   :", x * y)           # element-wise multiplication
print("x ** 2  :", x ** 2)          # square every element
print("sqrt(x) :", np.sqrt(x).round(3))  # square root of every element

# Scalar operations (apply one number to all elements)
print("\nx * 3   :", x * 3)
print("x + 100 :", x + 100)


# ----------------------------------------------------------
# 2.5 — Aggregation — summarising data
# ----------------------------------------------------------
print("\n--- Aggregation (summary statistics) ---")

scores = np.array([72, 88, 91, 65, 79, 95, 83, 70])
print("Test scores:", scores)
print("Mean (average):", scores.mean().round(2))
print("Min score     :", scores.min())
print("Max score     :", scores.max())
print("Sum           :", scores.sum())
print("Std deviation :", scores.std().round(2))   # how spread out the scores are

# Aggregation along axes in 2D arrays
matrix = np.array([[1, 2, 3],
                   [4, 5, 6]])
print("\nMatrix:\n", matrix)
print("Sum of each COLUMN (axis=0):", matrix.sum(axis=0))
print("Sum of each ROW    (axis=1):", matrix.sum(axis=1))

# 💬 DISCUSS: "When we compute accuracy in our model,
#              we're comparing two arrays and taking their mean!"


# ----------------------------------------------------------
# 2.6 — Indexing and Slicing
# ----------------------------------------------------------
print("\n--- Indexing and Slicing ---")

arr = np.array([10, 20, 30, 40, 50, 60, 70, 80])
print("Array:", arr)
print("First element  [0]   :", arr[0])
print("Last element   [-1]  :", arr[-1])
print("First 3        [:3]  :", arr[:3])
print("Last 3         [-3:] :", arr[-3:])
print("Middle         [2:5] :", arr[2:5])
print("Every other    [::2] :", arr[::2])

# 2D slicing
img = np.arange(25).reshape(5, 5)
print("\n5x5 image:\n", img)
print("Top-left 3x3 corner:\n", img[:3, :3])
print("Middle row (row index 2):", img[2, :])
print("Middle column (col 2):", img[:, 2])


# ----------------------------------------------------------
# 2.7 — Boolean Indexing — filter by condition
# ----------------------------------------------------------
print("\n--- Boolean Indexing (very useful!) ---")

pixel_values = np.array([23, 180, 45, 255, 12, 200, 89, 150])
print("Pixel values:", pixel_values)

# Which pixels are "bright" (> 100)?
bright_mask = pixel_values > 100
print("Bright mask (>100):", bright_mask)
print("Bright pixels only:", pixel_values[bright_mask])

# How many bright pixels?
print("Number of bright pixels:", bright_mask.sum())

# 💬 DISCUSS: "In image processing, we use this to find
#              specific regions of an image. Can you think of a use case?"


# ----------------------------------------------------------
# 2.8 — Speed Comparison: NumPy vs Plain Python
# ----------------------------------------------------------
print("\n--- SPEED TEST: NumPy vs Python ---")
import time

SIZE = 1_000_000   # 1 million numbers

# Plain Python
py_list = list(range(SIZE))
start = time.time()
py_result = [x * 2 for x in py_list]
py_time = time.time() - start

# NumPy
np_array = np.arange(SIZE)
start = time.time()
np_result = np_array * 2
np_time = time.time() - start

print(f"Plain Python list  : {py_time:.4f} seconds")
print(f"NumPy array        : {np_time:.4f} seconds")
print(f"NumPy is ~{py_time/np_time:.0f}x faster!")
print("\n💡 For 54,000 images with 150,000 pixels each,")
print("   this speed difference becomes enormous.")


# ============================================================
#  SECTION 3 — WHAT IS PYTORCH?
#  "NumPy's smarter cousin that can learn"
# ============================================================

print("\n" + "="*55)
print("SECTION 3: PyTorch — Tensors and Autograd")
print("="*55)

import torch


# ----------------------------------------------------------
# 3.1 — What is a Tensor?
# ----------------------------------------------------------
print("\n--- What is a Tensor? ---")

# A tensor is just like a NumPy array — but with superpowers:
#   ✅ Can run on a GPU (1000x faster for AI training)
#   ✅ Can automatically compute gradients (essential for learning)
#   ✅ Natively understood by PyTorch's neural network layers

# 0D tensor = a single number (scalar)
scalar = torch.tensor(42.0)
print("Scalar tensor:", scalar)
print("  Shape:", scalar.shape)       # torch.Size([]) — no dimensions

# 1D tensor = a vector
vector = torch.tensor([1.0, 2.0, 3.0, 4.0])
print("\n1D tensor:", vector)
print("  Shape:", vector.shape)       # torch.Size([4])

# 2D tensor = a matrix (like a grayscale image!)
matrix = torch.tensor([[1.0, 2.0, 3.0],
                       [4.0, 5.0, 6.0]])
print("\n2D tensor (matrix):\n", matrix)
print("  Shape:", matrix.shape)       # torch.Size([2, 3])

# 3D tensor = like a colour image (height x width x channels)
colour_image = torch.zeros(3, 64, 64)   # 3 channels, 64x64 pixels
print("\n3D tensor (colour image):")
print("  Shape:", colour_image.shape)   # torch.Size([3, 64, 64])

# 4D tensor = a BATCH of images (batch x channels x height x width)
batch = torch.zeros(32, 3, 64, 64)     # 32 images, 3 channels, 64x64
print("\n4D tensor (batch of 32 images):")
print("  Shape:", batch.shape)         # torch.Size([32, 3, 64, 64])

# 💬 DISCUSS: "When we feed images into our CNN,
#              they travel as 4D tensors. Now you know why!"


# ----------------------------------------------------------
# 3.2 — Creating Tensors (many ways)
# ----------------------------------------------------------
print("\n--- Creating Tensors ---")

# From a Python list
t1 = torch.tensor([10, 20, 30])
print("From list:", t1)

# Zeros and ones
t2 = torch.zeros(3, 4)
t3 = torch.ones(2, 3)
print("zeros(3,4):\n", t2)
print("ones(2,3):\n", t3)

# Random tensors (used to initialise neural network weights)
torch.manual_seed(42)
t4 = torch.rand(3, 3)           # uniform [0, 1]
t5 = torch.randn(3, 3)          # standard normal (mean=0, std=1)
print("\nrand(3,3):\n", t4.round(decimals=3))
print("\nrandn(3,3) — can be negative:\n", t5.round(decimals=3))

# Range
t6 = torch.arange(0, 10, 2)
print("\narange(0,10,2):", t6)


# ----------------------------------------------------------
# 3.3 — Tensor vs NumPy: converting between them
# ----------------------------------------------------------
print("\n--- Tensor ↔ NumPy Conversion ---")

np_arr = np.array([1.0, 2.0, 3.0])
tensor  = torch.from_numpy(np_arr)          # NumPy → Tensor
back    = tensor.numpy()                    # Tensor → NumPy

print("NumPy array :", np_arr)
print("As Tensor   :", tensor)
print("Back to NumPy:", back)

# 💬 DISCUSS: "We convert between them all the time —
#              model outputs → NumPy → matplotlib for plotting."


# ----------------------------------------------------------
# 3.4 — Tensor Math (same as NumPy!)
# ----------------------------------------------------------
print("\n--- Tensor Math ---")

a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])

print("a:", a)
print("b:", b)
print("a + b:", a + b)
print("a * b:", a * b)
print("a ** 2:", a ** 2)
print("torch.sqrt(a):", torch.sqrt(a).round(decimals=3))
print("dot product (a · b):", torch.dot(a, b))   # 1*4 + 2*5 + 3*6 = 32


# ----------------------------------------------------------
# 3.5 — The BIG difference: GPU support
# ----------------------------------------------------------
print("\n--- GPU Support ---")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device available:", device)

# On Kaggle/Colab with GPU:
#   device = cuda
# On a regular laptop:
#   device = cpu

# Move a tensor to the device (GPU or CPU)
tensor_on_device = torch.rand(3, 3).to(device)
print("Tensor device:", tensor_on_device.device)

# 💬 DISCUSS: "GPUs have thousands of tiny cores that can do
#              simple math in parallel. Perfect for tensors!"
print("\n💡 On a GPU, training that took 1 hour on CPU")
print("   can take just 5 minutes. Same code — just .to(device)!")


# ----------------------------------------------------------
# 3.6 — The BIGGEST difference: Autograd (automatic gradients)
# ----------------------------------------------------------
print("\n--- Autograd — How Neural Networks Learn ---")

# This is the magic that makes PyTorch special for AI.
# When requires_grad=True, PyTorch tracks every operation
# so it can compute how to update the weights later.

x = torch.tensor(3.0, requires_grad=True)
# Let's compute y = x^2 + 2x + 1
y = x**2 + 2*x + 1

print("x =", x)
print("y = x² + 2x + 1 =", y)

# Compute the gradient of y with respect to x
# (dy/dx = 2x + 2, so at x=3, dy/dx = 8)
y.backward()

print("Gradient dy/dx at x=3:", x.grad)
print("Expected: 2*3 + 2 =", 2*3 + 2, "✅")

# 💬 DISCUSS: "In training, y = the loss (how wrong we are).
#              x = the weights. The gradient tells us which direction
#              to move the weights to reduce the loss!"


# ----------------------------------------------------------
# 3.7 — A Mini Neural Network in PyTorch (preview!)
# ----------------------------------------------------------
print("\n--- A Tiny Neural Network (just a peek!) ---")
import torch.nn as nn

# The simplest possible neural network:
# Input (2 numbers) → Hidden layer (4 neurons) → Output (1 number)

class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(2, 4)    # 2 inputs → 4 neurons
        self.relu   = nn.ReLU()           # activation: negative → 0
        self.layer2 = nn.Linear(4, 1)    # 4 neurons → 1 output

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x

net   = TinyNet()
dummy = torch.tensor([[1.5, 0.8]])   # one sample with 2 features
out   = net(dummy)

print("Input:", dummy)
print("Output:", out)
print("\nNetwork structure:")
print(net)

total_params = sum(p.numel() for p in net.parameters())
print(f"\nTotal learnable parameters: {total_params}")
print("(These are the weights that training will adjust!)")


# ============================================================
#  SECTION 4 — NUMPY vs PYTORCH: SIDE-BY-SIDE
#  "They look almost the same — on purpose!"
# ============================================================

print("\n" + "="*55)
print("SECTION 4: NumPy vs PyTorch — Side by Side")
print("="*55)

print("""
  OPERATION            NUMPY                    PYTORCH
  ─────────────────────────────────────────────────────────
  Create array         np.array([1,2,3])        torch.tensor([1,2,3])
  Zeros                np.zeros((3,4))          torch.zeros(3,4)
  Ones                 np.ones((3,4))           torch.ones(3,4)
  Random               np.random.rand(3,4)      torch.rand(3,4)
  Shape                arr.shape                tensor.shape
  Reshape              arr.reshape(2,3)         tensor.reshape(2,3)
  Flatten              arr.flatten()            tensor.flatten()
  Mean                 arr.mean()               tensor.mean()
  Max                  arr.max()                tensor.max()
  Element multiply     arr1 * arr2              tensor1 * tensor2
  Dot product          np.dot(a, b)             torch.dot(a, b)
  To numpy             —                        tensor.numpy()
  From numpy           —                        torch.from_numpy(arr)
  GPU support          ❌ No                    ✅ Yes (.to(device))
  Auto gradients       ❌ No                    ✅ Yes (requires_grad)
  Neural network tools ❌ No                    ✅ Yes (nn.Module etc)
""")

# 💬 DISCUSS: "If they're so similar, why use both?"
print("💡 Answer:")
print("   NumPy  → data loading, preprocessing, plotting results")
print("   PyTorch → building, training, and running AI models")
print("   They work TOGETHER — like a team!")


# ============================================================
#  SECTION 5 — REAL AI PIPELINE PREVIEW
#  "Putting it all together — what happens when we train?"
# ============================================================

print("\n" + "="*55)
print("SECTION 5: Sneak Peek — NumPy + PyTorch in a Real Pipeline")
print("="*55)

import torch.optim as optim

# We'll simulate a tiny "is this leaf bright or dark?" classifier
# Input: one number = average brightness (0–255)
# Output: 0 = dark leaf (possibly diseased), 1 = bright (healthy)

torch.manual_seed(0)

# Fake training data — brightness values
# Labels: 1 = healthy (bright), 0 = diseased (dark)
np_brightness = np.array([210, 180, 160, 90, 50, 70, 200, 85], dtype=np.float32)
np_labels     = np.array([  1,   1,   1,  0,  0,  0,   1,  0], dtype=np.float32)

# Convert to PyTorch tensors
X = torch.from_numpy(np_brightness).unsqueeze(1)  # shape (8, 1)
y = torch.from_numpy(np_labels).unsqueeze(1)       # shape (8, 1)

# Define the simplest possible model: 1 input → 1 output
model     = nn.Linear(1, 1)
criterion = nn.BCEWithLogitsLoss()                 # binary classification loss
optimizer = optim.SGD(model.parameters(), lr=0.01)

print("Training a tiny brightness classifier...")
print("(Input = leaf brightness, Output = healthy/diseased)\n")

losses = []
for epoch in range(30):
    optimizer.zero_grad()
    output = model(X)
    loss   = criterion(output, y)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    if (epoch + 1) % 5 == 0:
        print(f"  Epoch {epoch+1:>2} | Loss: {loss.item():.4f}")

print("\n✅ Training complete!")

# Test on new values
test_brightness = torch.tensor([[230.0], [40.0]])
with torch.no_grad():
    predictions = torch.sigmoid(model(test_brightness))

print(f"\nBrightness 230 → {predictions[0].item():.2%} chance of healthy")
print(f"Brightness  40 → {predictions[1].item():.2%} chance of healthy")

# Plot loss curve using NumPy + Matplotlib
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 4))
plt.plot(np.arange(1, 31), losses, "g-o", markersize=4)
plt.title("Loss Curve — Tiny Brightness Classifier")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("\n💡 This tiny example uses the EXACT same pattern as Day 2's CNN!")
print("   torch.tensor → model → loss → backward → optimizer.step")
print("   That loop IS deep learning. You now understand it!")


# ============================================================
#  SECTION 6 — KEY TAKEAWAYS
# ============================================================

print("\n" + "="*55)
print("KEY TAKEAWAYS — Remember These!")
print("="*55)
print("""
  NumPy:
    ✅ Replaces slow Python loops with fast array operations
    ✅ Works on entire arrays at once (vectorisation)
    ✅ Foundation of data loading, preprocessing, and plotting
    ✅ Used everywhere in data science and AI

  PyTorch:
    ✅ Tensors = NumPy arrays + GPU + automatic gradients
    ✅ Autograd tracks operations to compute gradients automatically
    ✅ Gradients tell us how to update weights during training
    ✅ nn.Module = the building block of every neural network
    ✅ Same API as NumPy — easy to switch between them

  Together:
    ✅ NumPy handles data prep → PyTorch handles model training
    ✅ They pass data back and forth seamlessly
    ✅ Everything in our plant disease project uses BOTH

  Next up: EXERCISES!
    → Open Day1_Filler_EXERCISES.py and test what you learned!
""")
