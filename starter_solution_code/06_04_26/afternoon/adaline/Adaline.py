import torch

class Adaline1:
    def __init__ (self, num_features):
        self.num_fetures = num_features
        self.weights = torch.zeros((num_features, 1), dtype=torch.float32)
        self.bias = torch.zeros((1,), dtype=torch.float32)
    
    def forward(self, X):
        linear = torch.add(torch.mm(X, self.weights), self.bias)
        activation = linear
        return activation.view(-1)
    
    def backward(self, X, y_hat, y):
        
        grad_y_hat = 2 * (y_hat - y)
        grad_weights = X
        grad_bias = 1.
        
        g_w = torch.mm(grad_weights.t(), grad_y_hat.view(-1, 1)) / y.shape[0]
        g_b = torch.sum(grad_y_hat * grad_bias) / y.shape[0]
        
        return -1*g_w, -1*g_b
    
    def loss(self, y_hat, y):
        return torch.mean((y_hat - y) ** 2)
    
    def train(self, X, y, epochs=5, batch_size=10, learning_rate=0.01) -> list[float]:
        losses = []
        for e in range(epochs):
            
            torch.manual_seed(42)
            indices = torch.randperm(X.shape[0])
            X = X[indices]
            y = y[indices]
            
            mini_batches = torch.split(X, batch_size)
            y_mini_batches = torch.split(y, batch_size)
            
            for X_batch, y_batch in zip(mini_batches, y_mini_batches):
                
                y_hat = self.forward(X_batch)
                
                g_w, g_b = self.backward(X_batch, y_hat, y_batch)
                self.weights += learning_rate * g_w
                self.bias += learning_rate * g_b
                new_y_hat = self.forward(X_batch)
                # batch_loss = self.loss(new_y_hat, y_batch)
                # print(f"Epoch {e+1}, Batch Loss: {batch_loss.item():.4f}")
    
            epoch_loss = self.loss(self.forward(X), y)
            losses.append(epoch_loss.item())
            print(f"Epoch {e+1}, Epoch Loss: {epoch_loss.item():.4f}")
        return losses
    
    def evaluate(self, X, y):
        y_hat = self.forward(X)
        predictions = torch.where(y_hat > 0.5, 1., 0.)
        accuracy = torch.sum(predictions == y) / y.shape[0]
        return accuracy.item()
                   
                