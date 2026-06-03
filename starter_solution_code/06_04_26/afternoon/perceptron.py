import numpy as np;

class Perceptron:
    def __init__(self, num_features):
        self.num_features = num_features
        self.weights = np.zeros((num_features, 1), dtype=np.float64) ## increasing the dimensionality (num_of_features, 1)
        self.bias = np.zeros((1,), dtype=np.float64)
        
    def forward(self, X):
        linear = np.dot(X, self.weights) + self.bias
        return np.where(linear > 0., 1., 0.)
    
    def backward(self, X, y):
        prediction = self.forward(X)
        error = y - prediction
        return error
    
    def train(self, X, y, epochs=1):
        for e in range (epochs):
            for i in range(X.shape[0]):
                error = self.backward(X[i].reshape(1, self.num_features), y[i]).reshape(-1)
                self.weights += (error * X[i]).reshape(self.num_features, 1)
                self.bias += error
    
    def evaluate(self, X, y):    
        prediction = self.forward(X).reshape(-1)
        accuracy = np.sum(prediction == y) / y.shape[0]
        return accuracy