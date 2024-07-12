import numpy as np
from typing import List, Tuple

class AdvancedNeuralNetwork:
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int):
        self.layers = [input_dim] + hidden_dims + [output_dim]
        self.weights = [np.random.randn(self.layers[i], self.layers[i+1]) * np.sqrt(2 / self.layers[i]) 
                        for i in range(len(self.layers) - 1)]
        self.biases = [np.zeros((1, dim)) for dim in self.layers[1:]]
        self.velocities = [np.zeros_like(w) for w in self.weights]
        self.cache = {}

    def relu(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    def relu_derivative(self, x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, 1, 0)

    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def sigmoid_derivative(self, x: np.ndarray) -> np.ndarray:
        return x * (1 - x)

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.cache['A0'] = x
        for i in range(len(self.weights)):
            Z = np.dot(self.cache[f'A{i}'], self.weights[i]) + self.biases[i]
            self.cache[f'Z{i+1}'] = Z
            if i == len(self.weights) - 1:
                A = self.sigmoid(Z)
            else:
                A = self.relu(Z)
            self.cache[f'A{i+1}'] = A
        return A

    def backward(self, y: np.ndarray, learning_rate: float, momentum: float = 0.9) -> None:
        m = y.shape[0]
        dZ = self.cache[f'A{len(self.weights)}'] - y
        
        for i in reversed(range(len(self.weights))):
            dW = np.dot(self.cache[f'A{i}'].T, dZ) / m
            db = np.sum(dZ, axis=0, keepdims=True) / m
            
            self.velocities[i] = momentum * self.velocities[i] + learning_rate * dW
            self.weights[i] -= self.velocities[i]
            self.biases[i] -= learning_rate * db
            
            if i > 0:
                dA = np.dot(dZ, self.weights[i].T)
                dZ = dA * self.relu_derivative(self.cache[f'Z{i}'])

    def train(self, X: np.ndarray, y: np.ndarray, epochs: int, batch_size: int, learning_rate: float) -> List[float]:
        losses = []
        for epoch in range(epochs):
            for i in range(0, X.shape[0], batch_size):
                X_batch = X[i:i+batch_size]
                y_batch = y[i:i+batch_size]
                
                y_pred = self.forward(X_batch)
                self.backward(y_batch, learning_rate)
                
                loss = -np.mean(y_batch * np.log(y_pred) + (1 - y_batch) * np.log(1 - y_pred))
                losses.append(loss)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss}")
        
        return losses

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)

# Helper functions
def normalize_data(X: np.ndarray) -> np.ndarray:
    return (X - np.mean(X, axis=0)) / np.std(X, axis=0)

def train_test_split(X: np.ndarray, y: np.ndarray, test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    m = X.shape[0]
    test_size = int(m * test_size)
    indices = np.random.permutation(m)
    X_train, X_test = X[indices[test_size:]], X[indices[:test_size]]
    y_train, y_test = y[indices[test_size:]], y[indices[:test_size]]
    return X_train, X_test, y_train, y_test

# Usage example
if __name__ == "__main__":
    # Generate some random data
    np.random.seed(42)
    X = np.random.randn(1000, 10)
    y = (np.sum(X, axis=1) > 0).astype(float).reshape(-1, 1)

    # Normalize the data
    X = normalize_data(X)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    # Create and train the model
    model = AdvancedNeuralNetwork(input_dim=10, hidden_dims=[64, 32], output_dim=1)
    losses = model.train(X_train, y_train, epochs=100, batch_size=32, learning_rate=0.01)

    # Make predictions
    y_pred = model.predict(X_test)
    accuracy = np.mean((y_pred > 0.5) == y_test)
    print(f"Test Accuracy: {accuracy:.4f}")