import numpy as np

# ReLU activation function
def relu(x):
    return np.maximum(0, x)

# Single neuron forward pass
def neuron_forward(inputs, weights, bias):
    # Weighted sum + bias
    z = np.dot(weights, inputs) + bias
    # Activation
    return relu(z)

# Example: Vertical edge detector neuron
# Inputs = 4 pixels (2x2 image patch flattened)
inputs = np.array([0.1, 0.9, 0.1, 0.9])

# Weights = Vertical edge detector
weights = np.array([-1, +1, -1, +1])

# Bias
bias = 0.0

# Forward pass
output = neuron_forward(inputs, weights, bias)

print(f"Neuron output: {output}")
