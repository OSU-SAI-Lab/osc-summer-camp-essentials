import numpy as np

# The Perceptron is the simplest linear binary classifier.
# It learns a decision boundary (a line in 2D, a hyperplane in higher dims)
# that separates two classes (0 and 1) by iteratively adjusting weights
# whenever it makes a mistake.  No gradient descent — just a direct error
# correction rule: move the boundary toward the misclassified point.

class Perceptron:
    def __init__(self, num_features):
        self.num_features = num_features

        # Weights: one per input feature, stored as a column vector (n, 1)
        # so that matrix multiplication X @ weights gives a column of scores.
        # Initialized to all zeros — the perceptron starts with no preference
        # for any feature and learns purely from mistakes.
        self.weights = np.zeros((num_features, 1), dtype=np.float64)

        # A single bias term that shifts the decision boundary away from the
        # origin, giving the model more flexibility.
        self.bias = np.zeros((1,), dtype=np.float64)

    def forward(self, X):
        # Compute the linear (weighted) sum for every sample in X at once:
        #   linear[i] = w1*x1 + w2*x2 + ... + wn*xn + bias
        # Shape: (num_samples, 1)
        linear = np.dot(X, self.weights) + self.bias

        # Apply the Heaviside step function as the activation:
        #   output = 1  if linear > 0   (predict class 1)
        #          = 0  otherwise        (predict class 0)
        # This is what makes the Perceptron a hard binary classifier —
        # there is no "probability", just a yes/no decision.
        return np.where(linear > 0., 1., 0.)

    def backward(self, X, y):
        # Run the forward pass to get the current prediction
        prediction = self.forward(X)

        # Perceptron error signal: difference between the true label and prediction.
        #   error =  0  → correct prediction, no update needed
        #   error = +1  → predicted 0 but true label is 1 (false negative)
        #   error = -1  → predicted 1 but true label is 0 (false positive)
        # This signed error is used directly as the learning signal (no loss
        # function or gradient required).
        error = y - prediction
        return error

    def train(self, X, y, epochs=1):
        # Loop over the dataset multiple times (one full pass = one epoch).
        # More epochs give the model more chances to correct mistakes,
        # but for linearly separable data the Perceptron Convergence Theorem
        # guarantees it will eventually stop making errors.
        for e in range(epochs):
            # Online learning: process one sample at a time.
            # This is intentional — the Perceptron update rule is defined
            # per sample, not on batches.
            for i in range(X.shape[0]):
                # Reshape the single sample to (1, num_features) so forward()
                # treats it as a mini-batch of size 1.
                error = self.backward(
                    X[i].reshape(1, self.num_features), y[i]
                ).reshape(-1)  # flatten to scalar for the arithmetic below

                # Weight update rule:  w ← w + error * x
                # If error = 0: no change (correct prediction).
                # If error = +1: add x to weights, pulling the boundary
                #                toward classifying this point as 1.
                # If error = -1: subtract x, pushing the boundary away.
                self.weights += (error * X[i]).reshape(self.num_features, 1)

                # Bias update: same logic, but x is implicitly 1 for the bias.
                self.bias += error

    def evaluate(self, X, y):
        # Run the full dataset through the network and compare to true labels.
        prediction = self.forward(X).reshape(-1)  # flatten to 1-D for comparison

        # Fraction of samples where prediction matches the ground truth label.
        accuracy = np.sum(prediction == y) / y.shape[0]
        return accuracy
