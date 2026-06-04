import os
import torch as t
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn as nn
import matplotlib.pyplot as plt

# Path where trained weights are saved/loaded.
# If the file exists the script skips training entirely and goes straight to
# evaluation — re-run with FORCE_TRAIN=1 to retrain from scratch.
MODEL_PATH = './digit_net.pth'
FORCE_TRAIN = 0

# ── Data transforms ───────────────────────────────────────────────────────────

# transforms.Compose chains multiple preprocessing steps into one callable.
# Two steps are applied to every image before it enters the network:
#   1. ToTensor()   — converts a PIL image (H x W, uint8 0-255) into a
#                     float32 tensor of shape (C, H, W) with values in [0, 1]
#   2. Normalize((0.5,), (0.5,))
#                   — shifts the pixel range from [0, 1] to [-1, 1] using:
#                       pixel_out = (pixel_in - 0.5) / 0.5
#                     Centering inputs around 0 speeds up training because
#                     it keeps the initial activations from saturating.
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

# ── Dataset & DataLoader ──────────────────────────────────────────────────────

# MNIST is a classic benchmark: 70 000 grayscale images of handwritten digits
# (0-9), each 28×28 pixels.  torchvision downloads and caches it automatically.

# Training split: 60 000 images
mnist_trainset = datasets.MNIST(root='./data', train=True,  download=True, transform=transform)
# Test split:     10 000 images
mnist_testset  = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# DataLoader wraps the dataset in an iterable that yields mini-batches.
#   batch_size=10 → each iteration returns a tensor of shape (10, 1, 28, 28)
#   shuffle=True  → randomises sample order each epoch, which helps the model
#                   generalise and prevents it from memorising order patterns.
train_loader = t.utils.data.DataLoader(mnist_trainset, batch_size=10, shuffle=True)
test_loader  = t.utils.data.DataLoader(mnist_testset,  batch_size=10, shuffle=True)

# ── Model definition ──────────────────────────────────────────────────────────

# A simple fully-connected (dense) feed-forward network with three linear
# layers and ReLU activations between them.  All nn.Module subclasses must
# implement __init__ (register layers) and forward (define the computation).

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # Layer 1: 784 inputs → 100 hidden units
        # 28*28 = 784 because we flatten the 2-D image into a 1-D vector.
        self.linear1 = nn.Linear(28 * 28, 100)

        # Layer 2: 100 → 50 hidden units
        # Narrowing the width compresses the representation, forcing the
        # network to learn more abstract features.
        self.linear2 = nn.Linear(100, 50)

        # Output layer: 50 → 10 units, one per digit class (0-9).
        # No activation here — CrossEntropyLoss applies softmax internally.
        self.final = nn.Linear(50, 10)

        # ReLU (Rectified Linear Unit): f(x) = max(0, x)
        # Introduces non-linearity so the network can learn curved boundaries.
        # Without it, stacking linear layers collapses to a single linear map.
        self.relu = nn.ReLU()

    def forward(self, img):
        # img shape coming in: (batch_size, 1, 28, 28)

        # Flatten each image from (1, 28, 28) to (784,).
        # view(-1, 28*28) keeps the batch dimension and merges the rest.
        x = img.view(-1, 28 * 28)   # → (batch_size, 784)

        # Pass through layer 1 then apply ReLU: learn low-level features.
        x = self.relu(self.linear1(x))   # → (batch_size, 100)

        # Pass through layer 2 then apply ReLU: learn higher-level features.
        x = self.relu(self.linear2(x))   # → (batch_size, 50)

        # Final projection to class scores (called "logits").
        # Larger logit = model is more confident about that class.
        x = self.final(x)                # → (batch_size, 10)
        return x

net = Net()

# ── Train or load ─────────────────────────────────────────────────────────────

if os.path.exists(MODEL_PATH) and not FORCE_TRAIN:
    # Saved weights found — restore them directly into the model.
    # map_location='cpu' ensures the checkpoint loads even if it was saved on
    # a GPU machine but you're now running on CPU (and vice-versa).
    net.load_state_dict(t.load(MODEL_PATH, map_location='cpu'))
    print(f"Loaded weights from '{MODEL_PATH}' — skipping training.")
    print("  (To retrain from scratch, delete the file or run: FORCE_TRAIN=1 python digit.py)")

else:
    # ── Loss & optimiser ──────────────────────────────────────────────────────

    # CrossEntropyLoss = log-softmax + negative log-likelihood.
    # It expects raw logits (not softmax outputs) and integer class labels.
    # It penalises the model heavily when it is confident but wrong.
    cross_el = nn.CrossEntropyLoss()

    # Adam (Adaptive Moment Estimation) is a popular gradient-descent variant that
    # maintains a per-parameter learning rate, making it robust to noisy gradients.
    # lr=0.001 is a common default starting point.
    optimizer = t.optim.Adam(net.parameters(), lr=0.001)

    num_epochs = 10

    # ── Training loop ─────────────────────────────────────────────────────────

    for epoch in range(num_epochs):
        # net.train() switches the model to training mode.
        # This matters for layers like Dropout or BatchNorm (not used here, but
        # good practice to always set the mode explicitly).
        net.train()

        running_loss = 0.0
        for data in train_loader:
            x, y = data                    # x: (10, 1, 28, 28)  y: (10,)

            # Zero out gradients from the previous batch.
            # PyTorch accumulates gradients by default, so this must be called
            # before every backward pass to avoid mixing batches.
            optimizer.zero_grad()

            # Forward pass: compute logits for the current batch.
            output = net(x.view(-1, 28 * 28))   # → (10, 10)

            # Compute how wrong the predictions are.
            loss = cross_el(output, y)

            # Backward pass: compute d(loss)/d(weight) for every parameter.
            loss.backward()

            # Gradient descent step: nudge each weight in the direction that
            # reduces the loss.
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{num_epochs}  loss: {avg_loss:.4f}")

    # Save the learned weights so future runs skip training.
    # state_dict() is a plain dict of {layer_name: tensor} — the standard
    # PyTorch format for serialising model parameters.
    t.save(net.state_dict(), MODEL_PATH)
    print(f"\nWeights saved to '{MODEL_PATH}'.")

# ── Evaluation ────────────────────────────────────────────────────────────────

# net.eval() switches off training-only behaviour (Dropout, BatchNorm, etc.)
# Always call this before running inference or evaluation.
net.eval()

correct = 0
total   = 0

# t.no_grad() disables gradient tracking during evaluation.
# We don't need gradients here, so this saves memory and speeds things up.
with t.no_grad():
    for data in test_loader:
        x, y = data

        output = net(x.view(-1, 784))  # → (10, 10) logits

        for idx, i in enumerate(output):
            # t.argmax returns the index of the highest logit — the predicted
            # digit class.  Compare it to the true label y[idx].
            if t.argmax(i) == y[idx]:
                correct += 1
            total += 1

print(f'accuracy: {round(correct / total, 3)}')

# ── Visualisation ─────────────────────────────────────────────────────────────

data_point = 1

# Display the 4th image from the last test batch (0-indexed → index 3).
# x[3] shape is (1, 28, 28); view(28, 28) drops the channel dim for plotting.
plt.imshow(x[data_point].view(28, 28), cmap='gray')
plt.title("Sample test image")
plt.axis('off')
plt.show()

# Run that single image through the network and print the predicted digit.
# net(x[3].view(-1, 784)) returns shape (1, 10); [0] picks the first (only)
# row; t.argmax finds the class with the highest score.
print("Predicted digit:", t.argmax(net(x[data_point].view(-1, 784))[0]).item())
