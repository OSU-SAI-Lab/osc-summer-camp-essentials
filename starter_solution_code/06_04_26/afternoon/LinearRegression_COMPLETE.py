# ============================================================
#  🏠 LINEAR REGRESSION — COMPLETE TEACHING SCRIPT
#  Covers every concept from the slide deck:
#    function → y=wx+b → multiple variables →
#    house price problem → dataset → learning →
#    MSE → gradient descent → learning rate → epochs
#
#  No external dataset needed — we generate our own!
#  Run each SECTION one at a time and discuss as a class.
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

np.random.seed(42)
torch.manual_seed(42)

plt.rcParams.update({"figure.dpi": 130, "font.size": 12})

# ============================================================
#  SECTION 1 — WHAT IS A FUNCTION?
# ============================================================
print("=" * 60)
print("SECTION 1: What Is a Function?")
print("=" * 60)

# A function takes an INPUT and gives back an OUTPUT
# Let's write the simplest possible Python function

def double(x):
    """A function: give me x, I return 2x."""
    return 2 * x

print("\nf(x) = 2x")
print(f"  f(3)  = {double(3)}")
print(f"  f(10) = {double(10)}")

# 💬 DISCUSS: "Can anyone describe what this function does
#              in one plain English sentence?"

# ── The shopping function ──────────────────────────────────
print("\n── Shopping example: Total = w × x + b ──")

def shopping_total(x, w=2, b=1):
    """
    x = number of chocolates
    w = price per chocolate (£2)
    b = service charge (£1, always)
    """
    return w * x + b

for n in [1, 5, 10]:
    print(f"  Buy {n:>2} chocolates → £{shopping_total(n):.2f}")

# ── Multiple items ─────────────────────────────────────────
print("\n── Multiple items: Total = w1*x1 + w2*x2 + w3*x3 + b ──")

def basket_total(x1, x2, x3, w1=2.0, w2=1.5, w3=3.0, b=1.0):
    """
    x1 = chocolates (£2 each)
    x2 = biscuits   (£1.50 each)
    x3 = juice      (£3 each)
    b  = service charge (£1)
    """
    return w1*x1 + w2*x2 + w3*x3 + b

print(f"  3 choc, 2 biscuit, 1 juice → £{basket_total(3, 2, 1):.2f}")
print(f"  5 choc, 0 biscuit, 2 juice → £{basket_total(5, 0, 2):.2f}")

# 💬 DISCUSS: "In shopping we KNOW the prices (weights).
#              In machine learning — what if we don't know them?"


# ============================================================
#  SECTION 2 — THE HOUSE PRICE DATASET
#  We generate a realistic dataset that matches the slides
# ============================================================
print("\n" + "=" * 60)
print("SECTION 2: The House Price Dataset")
print("=" * 60)

N = 100   # number of houses

# True underlying weights (the model must DISCOVER these)
TRUE_W = np.array([2.8,    # size_m2:      $2,800 per m²
                   18.5,   # bedrooms:     $18,500 per bedroom
                  -12.3,   # dist_km:      -$12,300 per km from centre
                  -1.1,    # age_years:    -$1,100 per year of age
                   22.0])  # bathrooms:    $22,000 per bathroom
TRUE_B = 45.0              # base price:  $45,000

# Generate input features (realistic ranges)
size_m2   = np.random.randint(50,  220, N).astype(float)
bedrooms  = np.random.randint(1,   6,   N).astype(float)
dist_km   = np.round(np.random.uniform(0.5, 8.0, N), 1)
age_years = np.random.randint(1,   40,  N).astype(float)
bathrooms = np.random.randint(1,   4,   N).astype(float)

X_raw = np.column_stack([size_m2, bedrooms, dist_km, age_years, bathrooms])

# True price + small noise (real world is never perfect)
noise  = np.random.normal(0, 15, N)
prices = X_raw @ TRUE_W + TRUE_B + noise           # shape (100,)
prices = np.round(prices, 1)

# Build a nice DataFrame
df = pd.DataFrame({
    "size_m2":   size_m2,
    "bedrooms":  bedrooms,
    "dist_km":   dist_km,
    "age_years": age_years,
    "bathrooms": bathrooms,
    "price_k":   prices
})

print("\n📋 First 10 rows of our dataset:")
print(df.head(10).to_string(index=False))
print(f"\nDataset shape: {df.shape[0]} houses × {df.shape[1]} columns")
print(f"Price range:   ${prices.min():.0f}k  →  ${prices.max():.0f}k")

# Save CSV so students can open it in Excel too
df.to_csv("house_prices.csv", index=False)
print("\n💾 Saved as house_prices.csv")

# ── Quick visualisation ────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
fig.suptitle("Feature vs Price Relationships", fontsize=13, fontweight="bold")

for ax, col, color in zip(axes,
                          ["size_m2", "dist_km", "age_years"],
                          ["#065A82", "#E74C3C", "#7C5CBF"]):
    ax.scatter(df[col], df["price_k"], alpha=0.5, color=color, s=30)
    ax.set_xlabel(col)
    ax.set_ylabel("Price ($k)")
    ax.set_title(f"{col} vs Price")
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("feature_vs_price.png", bbox_inches="tight")
plt.show()
print("✅ Saved feature_vs_price.png")

# 💬 DISCUSS: "Look at the size plot. As size increases, what
#              happens to price? And for distance from centre?"


# ============================================================
#  SECTION 3 — THE PROBLEM: WE DON'T KNOW THE WEIGHTS
# ============================================================
print("\n" + "=" * 60)
print("SECTION 3: The Problem — Unknown Weights")
print("=" * 60)

print("""
  We have 100 houses with known features AND known prices.
  The real world function looks like:

    price = w1×size + w2×bedrooms + w3×dist + w4×age + w5×bath + b

  But what ARE w1, w2, w3, w4, w5, and b?

  In our shopping example, we knew them (w1=2, w2=1.5, w3=3).
  In the real world — nobody tells us.

  This is EXACTLY the machine learning problem:
    → Feed data in
    → Let the algorithm discover the weights
    → Use those weights to predict any new house

  Let's see what happens if we just GUESS the weights first.
""")

# Bad guess — all weights = 1.0, bias = 0
w_guess = np.ones(5)
b_guess = 0.0
y_guess = X_raw @ w_guess + b_guess
mse_guess = np.mean((y_guess - prices) ** 2)
print(f"  Random guess (all w=1, b=0): MSE = {mse_guess:,.0f}")
print(f"  A house worth ${prices[0]:.0f}k predicted as ${y_guess[0]:.0f}k")
print(f"  Error = ${abs(prices[0]-y_guess[0]):.0f}k\n")
print("  Clearly we need to LEARN the right weights.\n")


# ============================================================
#  SECTION 4 — NORMALISATION (why we need it)
# ============================================================
print("\n" + "=" * 60)
print("SECTION 4: Normalisation — Making the Numbers Friendly")
print("=" * 60)

print("""
  Our features have very different scales:
    size_m2   : 50 → 220     (large numbers)
    dist_km   : 0.5 → 8      (small numbers)
    age_years : 1 → 40       (medium numbers)

  If we train without normalising, the model will
  pay too much attention to large-scaled features.

  Normalisation = make all features roughly 0 to 1
  Formula: x_norm = (x - min) / (max - min)
""")

X_min  = X_raw.min(axis=0)
X_max  = X_raw.max(axis=0)
X_norm = (X_raw - X_min) / (X_max - X_min)

y_min  = prices.min()
y_max  = prices.max()
y_norm = (prices - y_min) / (y_max - y_min)

print("Feature ranges BEFORE normalisation:")
for name, mn, mx in zip(["size","beds","dist","age","bath"], X_min, X_max):
    print(f"  {name:6}: {mn:.1f} → {mx:.1f}")

print("\nFeature ranges AFTER normalisation:")
for name, mn, mx in zip(["size","beds","dist","age","bath"],
                         X_norm.min(axis=0), X_norm.max(axis=0)):
    print(f"  {name:6}: {mn:.2f} → {mx:.2f}")

print("\n✅ All features now between 0 and 1 — fair playing field!")


# ============================================================
#  SECTION 5 — NUMPY LINEAR REGRESSION (MANUAL GRADIENT DESCENT)
#  This is the "open the black box" version
# ============================================================
print("\n" + "=" * 60)
print("SECTION 5: Manual Gradient Descent — Step by Step")
print("=" * 60)

# Train / test split (80 / 20)
split     = int(0.8 * N)
X_train   = X_norm[:split]
y_train   = y_norm[:split]
X_test    = X_norm[split:]
y_test    = y_norm[split:]

n_train   = len(X_train)
n_feats   = X_train.shape[1]

print(f"\nTraining houses : {n_train}")
print(f"Test houses     : {len(X_test)}")
print(f"Features        : {n_feats}")

# ── Step 1: Initialise weights to zero ────────────────────
w = np.zeros(n_feats)
b = 0.0
print(f"\nStep 1 — Initial weights: {w}  bias: {b}")

# ── Step 2: One manual forward pass ───────────────────────
y_pred_init = X_train @ w + b
mse_init    = np.mean((y_pred_init - y_train) ** 2)
print(f"Step 2 — MSE before any learning: {mse_init:.6f}")

# ── Step 3: Compute gradients manually ────────────────────
errors  = y_pred_init - y_train          # shape (80,)
dw      = (2 / n_train) * X_train.T @ errors   # shape (5,)
db      = (2 / n_train) * errors.sum()

print(f"\nStep 3 — Gradients (which direction to move each weight):")
for i, (name, g) in enumerate(zip(["size","beds","dist","age","bath"], dw)):
    direction = "decrease w" if g > 0 else "increase w"
    print(f"  w{i+1} ({name:5}): gradient = {g:+.6f}  →  {direction}")

# ── Step 4: Update weights with small step ────────────────
lr = 0.1
w -= lr * dw
b -= lr * db

y_pred_after = X_train @ w + b
mse_after    = np.mean((y_pred_after - y_train) ** 2)
print(f"\nStep 4 — MSE after ONE update (lr={lr}): {mse_after:.6f}")
print(f"         MSE went from {mse_init:.6f} → {mse_after:.6f}  ✅ Improved!")

# 💬 DISCUSS: "If the gradient for w1 is positive, why do we
#              SUBTRACT it? What would happen if we ADDED it?"


# ============================================================
#  SECTION 6 — THE FULL TRAINING LOOP + EPOCHS
# ============================================================
print("\n" + "=" * 60)
print("SECTION 6: Full Training Loop — Epochs")
print("=" * 60)

def train_numpy(X_tr, y_tr, lr=0.1, epochs=500):
    """
    Full gradient descent training loop from scratch.
    Returns weights, bias, and loss history.
    """
    n, d = X_tr.shape
    w    = np.zeros(d)
    b    = 0.0
    history = []

    for epoch in range(epochs):
        y_pred = X_tr @ w + b
        mse    = np.mean((y_pred - y_tr) ** 2)
        history.append(mse)

        dw = (2/n) * X_tr.T @ (y_pred - y_tr)
        db = (2/n) * (y_pred - y_tr).sum()

        w -= lr * dw
        b -= lr * db

        if (epoch + 1) % 100 == 0:
            print(f"  Epoch {epoch+1:>4} | MSE: {mse:.6f}")

    return w, b, history

print(f"\nTraining with lr=0.1 for 500 epochs...")
w_trained, b_trained, loss_history = train_numpy(X_train, y_train, lr=0.1, epochs=500)

print(f"\n✅ Training complete!")
print(f"   Final MSE (normalised): {loss_history[-1]:.6f}")

# Plot loss curve
plt.figure(figsize=(9, 4))
plt.plot(loss_history, color="#065A82", lw=2)
plt.title("Loss Curve — MSE dropping over 500 Epochs", fontsize=13, fontweight="bold")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("loss_curve.png", bbox_inches="tight")
plt.show()
print("✅ Saved loss_curve.png")


# ============================================================
#  SECTION 7 — LEARNING RATE EXPERIMENTS
#  SHOW all three issues live: too large, just right, too small
# ============================================================
print("\n" + "=" * 60)
print("SECTION 7: Learning Rate Experiments 🔬")
print("=" * 60)

experiments = [
    {"lr": 5.0,    "label": "lr=5.0   — Too Large  (diverges!)", "color": "#E74C3C"},
    {"lr": 0.1,    "label": "lr=0.1   — Just Right (converges)", "color": "#27AE60"},
    {"lr": 0.0001, "label": "lr=0.0001 — Too Small  (barely learns)", "color": "#7C5CBF"},
]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Learning Rate: What Happens to the Loss Curve?", fontsize=13, fontweight="bold")

for exp in experiments:
    n, d  = X_train.shape
    w_exp = np.zeros(d)
    b_exp = 0.0
    hist  = []

    for epoch in range(200):
        y_p = X_train @ w_exp + b_exp
        mse = np.mean((y_p - y_train) ** 2)
        hist.append(min(mse, 5.0))         # cap for display
        dw = (2/n) * X_train.T @ (y_p - y_train)
        db = (2/n) * (y_p - y_train).sum()
        # Safety: skip update if gradients explode
        if np.any(np.abs(dw) > 1e6):
            hist.extend([5.0] * (200 - len(hist)))
            break
        w_exp -= exp["lr"] * dw
        b_exp -= exp["lr"] * db

    axes[0].plot(hist, color=exp["color"], lw=2, label=exp["label"])
    final_mse = hist[-1] if hist else 5.0
    print(f"  {exp['label']:<42}  Final MSE: {final_mse:.6f}")

axes[0].set_title("All 3 Learning Rates — 200 Epochs")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("MSE Loss")
axes[0].set_ylim(-0.02, 0.5)
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.3)

# Right panel: zoom in on just good LR vs bad
w_good = np.zeros(X_train.shape[1]); b_good = 0.0
w_bad  = np.zeros(X_train.shape[1]); b_bad  = 0.0
hist_good, hist_bad = [], []

for epoch in range(200):
    n = len(X_train)
    for (ww, bb, hist, lr_val) in [(w_good, b_good, hist_good, 0.1),
                                    (w_bad,  b_bad,  hist_bad,  0.0001)]:
        y_p = X_train @ ww + bb
        mse = np.mean((y_p - y_train) ** 2)
        hist.append(mse)
        dw = (2/n) * X_train.T @ (y_p - y_train)
        db = (2/n) * (y_p - y_train).sum()
        ww -= lr_val * dw
        bb -= lr_val * db

axes[1].plot(hist_good, color="#27AE60", lw=2, label="lr=0.1   — converging (good)")
axes[1].plot(hist_bad,  color="#7C5CBF", lw=2, label="lr=0.0001 — barely moving")
axes[1].set_title("Zoom: Good vs Too-Small LR")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("MSE Loss")
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("lr_experiment_live.png", bbox_inches="tight")
plt.show()
print("\n✅ Saved lr_experiment_live.png")

# 💬 DISCUSS: "Which learning rate would you choose and why?
#              What if your loss is oscillating up and down?"


# ============================================================
#  SECTION 8 — WHAT HAPPENS WITH MORE EPOCHS?
# ============================================================
print("\n" + "=" * 60)
print("SECTION 8: Epochs — How Many Is Enough?")
print("=" * 60)

epoch_checkpoints = [10, 50, 100, 200, 500, 1000]
results_table = []

for ep in epoch_checkpoints:
    n, d  = X_train.shape
    w_ep  = np.zeros(d)
    b_ep  = 0.0
    for _ in range(ep):
        y_p = X_train @ w_ep + b_ep
        dw  = (2/n) * X_train.T @ (y_p - y_train)
        db  = (2/n) * (y_p - y_train).sum()
        w_ep -= 0.1 * dw
        b_ep -= 0.1 * db
    train_mse = np.mean((X_train @ w_ep + b_ep - y_train)**2)
    test_mse  = np.mean((X_test  @ w_ep + b_ep - y_test )**2)
    results_table.append((ep, train_mse, test_mse))

print(f"\n{'Epochs':>8} {'Train MSE':>12} {'Test MSE':>12}  Note")
print("-" * 55)
for ep, tr, te in results_table:
    note = "still improving" if te > 0.005 else "converged ✅"
    print(f"{ep:>8} {tr:>12.6f} {te:>12.6f}  {note}")

# 💬 DISCUSS: "At what epoch does the model stop improving much?
#              Is there a point where training longer HURTS? (overfitting)"


# ============================================================
#  SECTION 9 — PYTORCH VERSION (same problem, autograd)
# ============================================================
print("\n" + "=" * 60)
print("SECTION 9: PyTorch Version — Autograd Does the Work")
print("=" * 60)

print("""
  The numpy version showed us EVERY calculation manually.
  PyTorch does the exact same thing but:
    → Computes gradients AUTOMATICALLY (autograd)
    → Can run on GPU for millions of parameters
    → Same training loop we'll use for CNNs

  The 5-line loop NEVER changes — only the model changes.
""")

# Convert to tensors
X_tr_t = torch.tensor(X_train, dtype=torch.float32)
y_tr_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_te_t = torch.tensor(X_test,  dtype=torch.float32)
y_te_t = torch.tensor(y_test,  dtype=torch.float32).unsqueeze(1)

# nn.Linear(5, 1) = w1*x1 + w2*x2 + ... + w5*x5 + b
model     = nn.Linear(5, 1)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

print("Model before training:")
print(f"  Weights: {model.weight.data.numpy().flatten().round(4)}")
print(f"  Bias   : {model.bias.data.item():.4f}")

pt_losses = []
EPOCHS = 500

print(f"\nTraining for {EPOCHS} epochs...\n")

for epoch in range(EPOCHS):
    # ── THE 5-LINE LOOP ───────────────────────────────────
    y_pred = model(X_tr_t)              # 1. Forward pass
    loss   = criterion(y_pred, y_tr_t)  # 2. Compute MSE
    optimizer.zero_grad()               # 3. Clear old gradients
    loss.backward()                     # 4. Compute new gradients
    optimizer.step()                    # 5. Update weights
    # ─────────────────────────────────────────────────────
    pt_losses.append(loss.item())

    if (epoch + 1) % 100 == 0:
        print(f"  Epoch {epoch+1:>4} | Loss: {loss.item():.6f}")

print("\n✅ PyTorch training complete!")
print(f"\nLearned weights (PyTorch):")
pt_weights = model.weight.data.numpy().flatten()
for name, w_val in zip(["size","beds","dist","age","bath"], pt_weights):
    print(f"  w ({name:5}): {w_val:.4f}")
print(f"  bias       : {model.bias.data.item():.4f}")

# Compare numpy vs PyTorch loss curves
plt.figure(figsize=(10, 4))
plt.plot(loss_history[:500], color="#065A82", lw=2, label="NumPy (manual)", alpha=0.8)
plt.plot(pt_losses,          color="#27AE60", lw=2, label="PyTorch (autograd)", linestyle="--")
plt.title("NumPy vs PyTorch — Same Result, Same Curve", fontsize=13, fontweight="bold")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("numpy_vs_pytorch.png", bbox_inches="tight")
plt.show()
print("✅ Saved numpy_vs_pytorch.png")


# ============================================================
#  SECTION 10 — MAKE REAL PREDICTIONS
# ============================================================
print("\n" + "=" * 60)
print("SECTION 10: Making Real Predictions")
print("=" * 60)

def predict_price(size, beds, dist, age, baths):
    """
    Predict house price using our trained PyTorch model.
    Inputs are in original units (m², count, km, years, count).
    """
    # Normalise using training min/max
    raw = np.array([[size, beds, dist, age, baths]], dtype=np.float32)
    norm = (raw - X_min) / (X_max - X_min)
    t    = torch.tensor(norm, dtype=torch.float32)
    with torch.no_grad():
        y_norm_pred = model(t).item()
    # De-normalise back to £k
    price = y_norm_pred * (y_max - y_min) + y_min
    return price

print("\n🏠 Predicting prices for new houses:\n")
test_houses = [
    {"size":100, "beds":3, "dist":2.0, "age":10, "baths":2, "label":"Average family home"},
    {"size":200, "beds":5, "dist":0.5, "age":2,  "baths":4, "label":"Large city-centre home"},
    {"size": 55, "beds":1, "dist":7.5, "age":35, "baths":1, "label":"Small old flat far out"},
]
for h in test_houses:
    pred = predict_price(h["size"], h["beds"], h["dist"], h["age"], h["baths"])
    print(f"  {h['label']}")
    print(f"    Size:{h['size']}m² | Beds:{h['beds']} | Dist:{h['dist']}km | "
          f"Age:{h['age']}yr | Baths:{h['baths']}")
    print(f"    → Predicted price: £{pred:.0f}k\n")

# 💬 DISCUSS: "Does the large city-centre home predict higher?
#              Does the old far-out flat predict lower? Do these make sense?"


# ============================================================
#  SECTION 11 — EVALUATING THE MODEL (R² + RESIDUAL PLOT)
# ============================================================
print("\n" + "=" * 60)
print("SECTION 11: Evaluating the Model — R² and Residuals")
print("=" * 60)

with torch.no_grad():
    y_pred_norm = model(X_te_t).numpy().flatten()

# De-normalise
y_pred_actual = y_pred_norm * (y_max - y_min) + y_min
y_true_actual = y_test      * (y_max - y_min) + y_min

# R² score
ss_tot = np.sum((y_true_actual - y_true_actual.mean()) ** 2)
ss_res = np.sum((y_true_actual - y_pred_actual)        ** 2)
r2     = 1 - ss_res / ss_tot

# RMSE (root mean squared error) — in same units as price
rmse = np.sqrt(np.mean((y_true_actual - y_pred_actual) ** 2))

print(f"\n  R²   = {r2:.4f}  (1.0 = perfect, 0.0 = no better than mean)")
print(f"  RMSE = £{rmse:.1f}k  (average prediction error)")

if r2 > 0.9:
    print("  → Excellent! Model explains >90% of price variation.")
elif r2 > 0.7:
    print("  → Good. Model explains most of price variation.")
else:
    print("  → Needs improvement. Try more features or more epochs.")

# Actual vs Predicted scatter plot
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Model Evaluation — Test Set", fontsize=13, fontweight="bold")

# Left: actual vs predicted
axes[0].scatter(y_true_actual, y_pred_actual, alpha=0.6, color="#065A82", s=40)
mn, mx = y_true_actual.min(), y_true_actual.max()
axes[0].plot([mn, mx], [mn, mx], "r--", lw=1.5, label="Perfect prediction")
axes[0].set_xlabel("Actual Price (£k)")
axes[0].set_ylabel("Predicted Price (£k)")
axes[0].set_title(f"Actual vs Predicted  (R² = {r2:.3f})")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Right: residuals (errors)
residuals = y_true_actual - y_pred_actual
axes[1].scatter(y_pred_actual, residuals, alpha=0.6, color="#E74C3C", s=40)
axes[1].axhline(0, color="black", lw=1.5, linestyle="--")
axes[1].set_xlabel("Predicted Price (£k)")
axes[1].set_ylabel("Residual (Actual − Predicted)")
axes[1].set_title("Residual Plot\n(dots near 0 = good predictions)")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("model_evaluation.png", bbox_inches="tight")
plt.show()
print("✅ Saved model_evaluation.png")

# 💬 DISCUSS: "What does the residual plot tell us?
#              If the dots form a pattern (not random),
#              what might that mean?"


# ============================================================
#  SECTION 12 — PUTTING IT ALL TOGETHER
# ============================================================
print("\n" + "=" * 60)
print("SECTION 12: Summary — The Complete Picture")
print("=" * 60)
print(f"""
  WHAT WE BUILT:
    ✅ A dataset of {N} houses with 5 features + price
    ✅ Normalised features for fair training
    ✅ Manual gradient descent (numpy) — saw every calculation
    ✅ PyTorch gradient descent — autograd handles it
    ✅ Predicted prices for new unseen houses
    ✅ Evaluated with R² = {r2:.3f} and RMSE = £{rmse:.1f}k

  CONCEPTS YOU SAW IN ACTION:
    📐 y = w₁x₁ + w₂x₂ + ... + w₅x₅ + b
    🎲 Initial weights = 0 (random guess)
    📏 MSE = how wrong we are after each prediction
    🧭 Gradient = which direction to adjust each weight
    🐢 Learning rate = how big each adjustment step is
    🔁 Epoch = one full pass through all training data
    📈 Loss curve = proof that learning is happening

  THE TRAINING LOOP (NEVER CHANGES):
    for epoch in range(epochs):
        y_pred = model(X)           ← forward pass
        loss   = criterion(y, pred) ← MSE
        zero_grad()                 ← clear old gradients
        loss.backward()             ← compute new gradients
        optimizer.step()            ← update weights

  This same loop runs ResNet18, GPT, and every other AI model.
""")
