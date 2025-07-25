import numpy as np

def relu(x):
    return np.maximum(0, x)

def forward_pass(x, weights1, bias1, weights2, bias2):
    # First layer
    z1 = np.dot(weights1, x) + bias1
    a1 = relu(z1)

    # Second layer
    z2 = np.dot(weights2, a1) + bias2
    a2 = relu(z2)

    return a2  # Output of the final layer (predicted value)

# Input (4 features)
x = np.array([0.2, 0.4, 0.6, 0.8])

# Layer 1: 3 neurons
weights1 = np.array([
    [0.5, -0.6, 0.2, 0.1],
    [-0.3, 0.8, -0.5, 0.4],
    [0.2, 0.1, 0.9, -0.7]
])
bias1 = np.array([0.1, 0.2, 0.0])

# Layer 2: 2 neurons (e.g. cat vs not-cat)
weights2 = np.array([
    [0.3, -0.2, 0.5],
    [-0.6, 0.1, 0.4]
])
bias2 = np.array([0.05, -0.1])

# Forward pass
output = forward_pass(x, weights1, bias1, weights2, bias2)

print("Network output:", output)