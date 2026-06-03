import numpy as np
import matplotlib.pyplot as plt

from thrusday.perceptron import Perceptron

## Constants
TRAIN_TEST_SPLIT = 0.8
SEED_VALUE = 42
EPOCHS = 4

## all numbers will stick to float64 unless specified as some of the features as float
data = np.genfromtxt('/Users/brijeshnanda/projects/deep-learning-tutorials/perceptron/data/perceptron_toydata.txt', delimiter='\t')

## Data extraction

X = data[:, :-1]
Y = data[:, -1]

print("Shape of X:", X.shape)
print("Shape of Y:", Y.shape)

num_features = X.shape[1]
data_size = X.shape[0]

## Train test split:: this approach sets the global seed, 
# np.random.seed(SEED_VALUE)
shuffle_rng = np.random.default_rng(seed=SEED_VALUE)
indices = np.arange(data_size)
shuffle_rng.shuffle(indices) ## this return none

split_index = int(data_size * TRAIN_TEST_SPLIT)

X_Train = X[indices[:split_index], :]
Y_Train = Y[indices[:split_index]]

X_Test = X[indices[split_index:], :]
Y_Test = Y[indices[split_index:]]

## normalization

mu, sigma = np.mean(X_Train, axis=0), np.std(X_Train, axis=0)
X_Train = (X_Train - mu) / sigma
X_Test = (X_Test - mu) / sigma

## visualization of data points
### Train 

plt.scatter(X_Train[:, 0], X_Train[:, 1], c=Y_Train, cmap='bwr', edgecolors='k')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Training Data')
plt.show()

### Test

plt.scatter(X_Test[:, 0], X_Test[:, 1], c=Y_Test, cmap='bwr', edgecolors='k')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Test Data')
plt.show()

## Model Train

model = Perceptron(num_features)
model.train(X_Train, Y_Train, epochs=EPOCHS)

model_weights = model.weights
model_bias = model.bias
print("Model weights:\n", model_weights)
print("Model bias:\n", model_bias)

## Model Evaluation
train_accuracy = model.evaluate(X_Train, Y_Train)
test_accuracy = model.evaluate(X_Test, Y_Test)
print("Train Accuracy:", train_accuracy*100, "%")
print("Test Accuracy:", test_accuracy*100, "%")

x_min = X_Test[:, 0].min() - 1
x_max = X_Test[:, 0].max() + 1
y_min = - (model_bias + model_weights[0] * x_min).item() / model_weights[1].item()
y_max = - (model_bias + model_weights[0] * x_max).item() / model_weights[1].item() 

# x1 = (-model_bias / model_weights[0]).item()
# x2 = (-model_bias / model_weights[1]).item()
# print("Decision boundary points: x1 =", x1, ", x2 =", x2)
x1_values = [x_min, x_max]
x2_values = [y_min, y_max]
plt.plot(x1_values, x2_values, 'k', label='Decision Boundary')

plt.scatter(X_Test[:, 0], X_Test[:, 1], c=Y_Test, cmap='bwr', edgecolors='k')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Test Data with Decision Boundary')
plt.show()

