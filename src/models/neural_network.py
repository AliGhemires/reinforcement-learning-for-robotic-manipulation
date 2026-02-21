import numpy as np
from typing import Tuple, List, Any


class NeuralNetwork:
    """A simple feedforward neural network for function approximation."""
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int, learning_rate: float):
        """Initializes the neural network parameters."""
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        # Weight initialization using He initialization for ReLU
        self.weights_input_hidden = np.random.randn(input_size, hidden_size) * np.sqrt(2. / input_size)
        self.weights_hidden_output = np.random.randn(hidden_size, output_size) * np.sqrt(2. / hidden_size)
        self.bias_hidden = np.zeros((1, hidden_size))
        self.bias_output = np.zeros((1, output_size))

        # Activation functions
        self.sigmoid = lambda x: 1 / (1 + np.exp(-x))
        self.relu = lambda x: np.maximum(0, x)
        self.softmax = lambda x: np.exp(x - np.max(x, axis=1, keepdims=True)) / np.sum(np.exp(x - np.max(x, axis=1, keepdims=True)), axis=1, keepdims=True)

    def feedforward(self, x: np.ndarray) -> np.ndarray:
        """Computes the forward pass of the network."""
        if x.shape[1] != self.input_size:
            raise ValueError(f"Input shape {x.shape[1]} must match input_size {self.input_size}.")

        # Input to hidden layer
        z_hidden = np.dot(x, self.weights_input_hidden) + self.bias_hidden
        a_hidden = self.relu(z_hidden)

        # Hidden layer to output layer
        z_output = np.dot(a_hidden, self.weights_hidden_output) + self.bias_output
        output = self.softmax(z_output)
        return output

    def backpropagate(self, x: np.ndarray, y: np.ndarray, output: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Computes backward pass and updates weights."""
        n_samples = x.shape[0]

        # Validate output shape
        if output.shape != y.shape:
            raise ValueError("Output shape must match target shape.\nGot output shape {} and target shape {}.".format(output.shape, y.shape))

        # Calculate loss gradient
        delta_output = output - y
        grad_weights_hidden_output = np.dot(self.relu(np.dot(x, self.weights_input_hidden) + self.bias_hidden).T, delta_output) / n_samples
        grad_bias_output = np.sum(delta_output, axis=0, keepdims=True) / n_samples

        # Backpropagate to hidden layer
        delta_hidden = np.dot(delta_output, self.weights_hidden_output.T) * (np.dot(x, self.weights_input_hidden) + self.bias_hidden > 0)  # ReLU derivative
        grad_weights_input_hidden = np.dot(x.T, delta_hidden) / n_samples
        grad_bias_hidden = np.sum(delta_hidden, axis=0, keepdims=True) / n_samples

        # Update weights and biases
        self.weights_input_hidden -= self.learning_rate * grad_weights_input_hidden
        self.bias_hidden -= self.learning_rate * grad_bias_hidden
        self.weights_hidden_output -= self.learning_rate * grad_weights_hidden_output
        self.bias_output -= self.learning_rate * grad_bias_output

    def train(self, x: np.ndarray, y: np.ndarray, epochs: int):
        """Train the neural network using provided data."""
        for epoch in range(epochs):
            output = self.feedforward(x)
            self.backpropagate(x, y, output)
            if epoch % 100 == 0:
                loss = self.compute_loss(output, y)
                print(f'Epoch {epoch}, Loss: {loss:.4f}')  # Added precision for better insight

    def compute_loss(self, output: np.ndarray, y: np.ndarray) -> float:
        """Compute cross-entropy loss between predicted and true values."""
        n_samples = y.shape[0]
        # Using log-sum-exp for numerical stability
        return -np.sum(y * np.log(output + 1e-10)) / n_samples  # Add epsilon for numerical stability


if __name__ == '__main__':
    # Example usage
    nn = NeuralNetwork(input_size=3, hidden_size=5, output_size=2, learning_rate=0.01)
    # Dummy input and output
    X = np.random.rand(10, 3)  # 10 samples, 3 features
    Y = np.random.rand(10, 2)  # Ensure Y is properly normalized when used in actual application
    nn.train(X, Y, epochs=1000)
