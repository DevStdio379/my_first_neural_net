import numpy as np

# ReLU activation function
def relu(x):
    return np.maximum(0, x)

# Each row in weights = one neuron
def layer_forward(input, weights, biases):
    # Weighted sum + bias
    z = np.dot(weights, input) + biases
    # Activation
    return relu(z)

# Inputs (e.g. 4 pixels from image patch)
inputs = np.array([0.1, 0.9, 0.1, 0.9])

# Layer with 3 neurons → 3 filters (e.g. vertical, horizontal, diagonal)
weights = np.array([
    [-1, 1, -1, 1],         # Neuron 1: vertical edge detector
    [1, 1, -1, -1],         # Neuron 2: horizontal edge
    [0.5, -0.5, 0.5, -0.5]  # Neuron 3: diagonal
])

# Biases
biases = np.array([0, 0, 0])

# Forward pass
outputs = layer_forward(inputs, weights, biases)

print(f"Neuron output: {outputs}")