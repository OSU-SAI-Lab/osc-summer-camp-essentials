import numpy as np
import matplotlib.pyplot as plt

# Import our Perceptron class from the sibling module
from perceptron import Perceptron

# ── Hyperparameters / constants ───────────────────────────────────────────────

TRAIN_TEST_SPLIT = 0.8   # 80 % of samples go to training, 20 % held out for testing
SEED_VALUE = 42          # Fixed seed makes the shuffle reproducible across runs
EPOCHS = 4               # Number of full passes over the training data

# ── Load data ─────────────────────────────────────────────────────────────────

# np.genfromtxt reads a tab-delimited text file into a 2-D NumPy array.
# dtype defaults to float64, so every value is stored as a 64-bit float.
data = np.genfromtxt('data/perceptron_toydata.txt', delimiter='\t')

# ── Split into features (X) and labels (Y) ───────────────────────────────────

# By convention the last column is the class label (0 or 1).
# All other columns are input features.
X = data[:, :-1]   # shape: (num_samples, num_features)
Y = data[:, -1]    # shape: (num_samples,)

print("Shape of X:", X.shape)
print("Shape of Y:", Y.shape)

num_features = X.shape[1]   # number of input dimensions
data_size    = X.shape[0]   # total number of data points

# ── Shuffle & split ───────────────────────────────────────────────────────────

# We shuffle *indices* rather than the array itself so we can apply the same
# permutation to both X and Y without misaligning them.
#
# np.random.default_rng is the modern NumPy API: it creates an isolated
# random-number generator instead of mutating the global state, which makes
# the code easier to reason about in larger projects.
shuffle_rng = np.random.default_rng(seed=SEED_VALUE)
indices = np.arange(data_size)
shuffle_rng.shuffle(indices)  # shuffle in-place; returns None

# The first 80 % of shuffled indices become training samples.
split_index = int(data_size * TRAIN_TEST_SPLIT)

X_Train = X[indices[:split_index], :]   # (train_size, num_features)
Y_Train = Y[indices[:split_index]]      # (train_size,)

X_Test  = X[indices[split_index:], :]  # (test_size, num_features)
Y_Test  = Y[indices[split_index:]]     # (test_size,)

# ── Z-score normalization ─────────────────────────────────────────────────────

# Normalize so that each feature has mean ≈ 0 and std ≈ 1.
# This prevents features with large numeric ranges from dominating the weight
# updates and makes training more stable.
#
# IMPORTANT: compute mu and sigma on training data only, then apply the same
# statistics to the test set.  Using test statistics would constitute data
# leakage — the model would "see" test information before evaluation.
mu    = np.mean(X_Train, axis=0)   # per-feature mean,  shape: (num_features,)
sigma = np.std(X_Train,  axis=0)   # per-feature std,   shape: (num_features,)

X_Train = (X_Train - mu) / sigma
X_Test  = (X_Test  - mu) / sigma   # use train mu/sigma here — not test stats

# ── Visualize raw data ────────────────────────────────────────────────────────

# Scatter plot colors each point by its class label (Y).
# cmap='bwr' maps 0 → blue, 1 → red; edgecolors='k' adds a black outline
# so points are visible even when two colors overlap.

# Training set
plt.scatter(X_Train[:, 0], X_Train[:, 1], c=Y_Train, cmap='bwr', edgecolors='k')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Training Data')
plt.show()

# Test set
plt.scatter(X_Test[:, 0], X_Test[:, 1], c=Y_Test, cmap='bwr', edgecolors='k')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Test Data')
plt.show()

# ── Train the model ───────────────────────────────────────────────────────────

model = Perceptron(num_features)
model.train(X_Train, Y_Train, epochs=EPOCHS)

# Inspect the learned parameters to understand what the model found
model_weights = model.weights
model_bias    = model.bias
print("Model weights:\n", model_weights)
print("Model bias:\n",    model_bias)

# ── Evaluate accuracy ─────────────────────────────────────────────────────────

train_accuracy = model.evaluate(X_Train, Y_Train)
test_accuracy  = model.evaluate(X_Test,  Y_Test)
print("Train Accuracy:", train_accuracy * 100, "%")
print("Test Accuracy:",  test_accuracy  * 100, "%")

# ── Visualize decision boundary ───────────────────────────────────────────────

# The decision boundary is the set of points where the linear output = 0:
#   w0*x1 + w1*x2 + bias = 0
# Solving for x2:
#   x2 = -(bias + w0*x1) / w1
#
# We evaluate this at the leftmost and rightmost x1 values on the plot to get
# two endpoints of the boundary line, then draw a straight line between them.

x_min = X_Test[:, 0].min() - 1   # a little left of the data
x_max = X_Test[:, 0].max() + 1   # a little right of the data

# Solve x2 = -(bias + w0 * x1) / w1 at both endpoints
y_min = -(model_bias + model_weights[0] * x_min).item() / model_weights[1].item()
y_max = -(model_bias + model_weights[0] * x_max).item() / model_weights[1].item()

x1_values = [x_min, x_max]
x2_values = [y_min, y_max]

# Plot the boundary on top of the test scatter
plt.plot(x1_values, x2_values, 'k', label='Decision Boundary')
plt.scatter(X_Test[:, 0], X_Test[:, 1], c=Y_Test, cmap='bwr', edgecolors='k')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Test Data with Decision Boundary')
plt.legend()
plt.show()
