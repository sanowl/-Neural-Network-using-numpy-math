import numpy as np

class NeuralNetwork:
  def __init__(self, input_dim, hidden_dims, output_dim):
    self.weights = [2 * np.random.random((input_dim, hidden_dims[0])) - 1]
    for i in range(len(hidden_dims) - 1):
      self.weights.append(2 * np.random.random((hidden_dims[i], hidden_dims[i+1])) - 1)
    self.weights.append(2 * np.random.random((hidden_dims[-1], output_dim)) - 1)
    self.momentum = [np.zeros_like(w) for w in self.weights]
    self.velocity = [np.zeros_like(w) for w in self.weights]

  def sigmoid(self, x):
    return 1 / (1 + np.exp(-x))

  def sigmoid_derivative(self, x):
    return x * (1 - x)

  def tanh(self, x):
    return np.tanh(x)

  def tanh_derivative(self, x):
    return 1 - np.tanh(x) ** 2

  def relu(self, x):
    return np.maximum(0, x)

  def relu_derivative(self, x):
    x[x <= 0] = 0
    x[x > 0] = 1
    return x

  def dropout(self, X, dropout_rate):
    keep_prob = 1 - dropout_rate
    mask = np.random.uniform(0, 1.0, X.shape) < keep_prob
    scale = (1/keep_prob) if keep_prob > 0.0 else 0.0
    return mask * X * scale

  def batch_norm(self, X, gamma=1, beta=0, eps=1e-5):
    mean = np.mean(X, axis=0)
    var = np.var(X, axis=0)
    X_norm = (X - mean) / np.sqrt(var + eps)
    out = gamma * X_norm + beta
    return out

  def update_weights_adam(self, weights, gradients, learning_rate, beta1, beta2, epsilon, t):
    m = beta1*self.momentum[weights] + (1-beta1)*gradients
    v = beta2*self.velocity[weights] + (1-beta2)*(gradients**2)
    m_hat = m / (1-beta1**t)
    v_hat = v / (1-beta2**t)
    self.weights[weights] -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
    self.momentum[weights] = m
    self.velocity[weights] = v

  def forward_propagation(self, inputs, dropout_rate):
    layer_1 = self.sigmoid(np.dot(inputs, self.weights[0]))
    layer_1 = self.dropout(layer_1, dropout_rate)
    layer_1 = self.batch_norm(layer_1)
    layer_2 = self.tanh(np.dot(layer_1, self.weights[1]))
    layer_2 = self.dropout(layer_2, dropout_rate)
    layer_2 = self.batch_norm(layer_2)
    layer_3 = self.relu(np.dot(layer_2, self.weights[2]))
    layer_3 = self.dropout(layer_3, dropout_rate)
    layer_3 = self.batch_norm(layer_3)
    output = self.sigmoid(np.dot(layer_3, self.weights[2]))
    return output, layer_3, layer_2, layer_1

  def train(self, inputs, expected_outputs, epochs, learning_rate, dropout_rate, batch_size, beta1, beta2, epsilon):
    for epoch in range(epochs):
      for batch_start in range(0, len(inputs), batch_size):
        batch_end = min(batch_start + batch_size, len(inputs))
        batch_inputs = inputs[batch_start:batch_end]
        batch_expected_outputs = expected_outputs[batch_start:batch_end]
        output, layer_3, layer_2, layer_1 = self.forward_propagation(batch_inputs, dropout_rate)
        error = batch_expected_outputs - output
        delta_output = error * self.sigmoid_derivative(output)
        error_layer_3 = delta_output.dot(self.weights[2].T)
        delta_layer_3 = error_layer_3 * self.relu_derivative(layer_3)
        error_layer_2 = delta_layer_3.dot(self.weights[1].T)
        delta_layer_2 = error_layer_2 * self.tanh_derivative(layer_2)
        error_layer_1 = delta_layer_2.dot(self.weights[0].T)
        delta_layer_1 = error_layer_1 * self.sigmoid_derivative(layer_1)
        self.update_weights_adam(0, delta_layer_1, learning_rate, beta1, beta2, epsilon, epoch+1)
        self.update_weights_adam(1, delta_layer_2, learning_rate, beta1, beta2, epsilon, epoch+1)
        self.update_weights_adam(2, delta_layer_3, learning_rate, beta1, beta2, epsilon, epoch+1)

  def predict(self, inputs):
    output, _, _, _ = self.forward_propagation(inputs, 0)
    return output