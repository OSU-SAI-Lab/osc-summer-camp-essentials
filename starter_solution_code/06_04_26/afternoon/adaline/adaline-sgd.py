import torch
import pandas as pd
import matplotlib.pyplot as plt

from Adaline import Adaline1

torch.manual_seed(42)
TRAIN_TEST_SPLIT = 0.7

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

data = pd.read_csv('./datasets/iris.data', delimiter=',', header=None, names=['x1', 'x2', 'x3', 'x4', 'y'])

print(data.head())
data = data.iloc[50:150]
print("size of data:", data.shape[0])
X = data.iloc[:, 2:4].values
y = data.iloc[:, -1].values

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor([0 if label == 'Iris-versicolor' else 1 for label in y], dtype=torch.float32)

print("Shape of X_tensors:", X.shape)
print("Shape of y_tensors:", y.shape) 

print("Unique classes in y:", torch.unique(y))

plt.scatter(X[y == 1, 0], X[y == 1, 1], c='red', marker='o', label='Iris-versicolor')
plt.scatter(X[y == 0, 0], X[y == 0, 1], c='blue', marker='x', label='Other')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Iris Dataset - Binary Classification')
plt.legend()
plt.show()

## data shuffle 
torch.manual_seed(42)
indices = torch.randperm(X.shape[0])
print("Shuffled indices:", indices)
print("split", int(TRAIN_TEST_SPLIT * len(indices)))
X_tensors = X[indices[:int(TRAIN_TEST_SPLIT * len(indices))]]
y_tensors = y[indices[:int(TRAIN_TEST_SPLIT * len(indices))]]

print("done")
X_test_tensors = X[indices[int(TRAIN_TEST_SPLIT * len(indices)):]]
y_test_tensors = y[indices[int(TRAIN_TEST_SPLIT * len(indices)):]]

## z-score normalization
mu, sigma = X_tensors.mean(dim=0), X_tensors.std(dim=0)
X_tensors = (X_tensors - mu) / sigma
X_test_tensors = (X_test_tensors - mu) / sigma

## model initialization

adaline = Adaline1(num_features=X_tensors.shape[1])
losses = adaline.train(X_tensors, y_tensors, epochs=200, batch_size=6, learning_rate=0.01)
plt.plot(range(len(losses)), losses)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Adaline Training Loss')
plt.show()

## evaluation 
test_accuracy = adaline.evaluate(X_test_tensors, y_test_tensors)
print("Test Accuracy:", test_accuracy*100, "%")
train_Accuracy = adaline.evaluate(X_tensors, y_tensors)
print("Train Accuracy:", train_Accuracy*100, "%")

### Decision boundary (train data)

## weights and bias
weights = adaline.weights
bias = adaline.bias - 0.5

x1_min = X_tensors[:, 0].min() - 1
x1_max = X_tensors[:, 0].max() + 1


X2_min = - (bias + weights[0] * x1_min) / weights[1]
X2_max = - (bias + weights[0] * x1_max) / weights[1]
x1_values = [x1_min, x1_max]
x2_values = [X2_min, X2_max]

plt.scatter(X_tensors[y_tensors == 1, 0], X_tensors[y_tensors == 1, 1], c='red', marker='o', label='Iris-versicolor')
plt.scatter(X_tensors[y_tensors == 0, 0], X_tensors[y_tensors == 0, 1], c='blue', marker='x', label='Other')
plt.plot(x1_values, x2_values, 'k', label='Decision Boundary')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Adaline Decision Boundary')
plt.legend()
plt.show()

### Decision boundary (test data)

x1_min = X_test_tensors[:, 0].min() - 1
x1_max = X_test_tensors[:, 0].max() + 1


X2_min = - (bias + weights[0] * x1_min).item() / weights[1].item()
X2_max = - (bias + weights[0] * x1_max).item() / weights[1].item()
x1_values = [x1_min, x1_max]
x2_values = [X2_min, X2_max]

plt.scatter(X_test_tensors[y_test_tensors == 1, 0], X_test_tensors[y_test_tensors == 1, 1], c='red', marker='o', label='Iris-versicolor')
plt.scatter(X_test_tensors[y_test_tensors == 0, 0], X_test_tensors[y_test_tensors == 0, 1], c='blue', marker='x', label='Other')
plt.plot(x1_values, x2_values, 'k', label='Decision Boundary')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Adaline Decision Boundary')
plt.legend()
plt.show()







