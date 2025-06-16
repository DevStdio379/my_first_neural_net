import numpy as np

# ReLU activation function for hidden layer
def relu(x):
    return np.maximum(0, x)

# Sigmoid activation function for output layer
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Binary cross entropy for loss calculation
def binary_cross_entropy(pred, target):
    return -(target * np.log(pred + 1e-8) + (1 - target) * np.log(1 - pred + 1e-8))

# Example input (4 features: like simplified pixels)
X = np.array([
    [0, 1, 0, 1],   # cat-like
    [1, 0, 1, 0],   # not-cat
    [0.5, 1, 0.5, 1],  # cat-like
    [1, 0.2, 1, 0.1]   # not-cat
])

# Labels: 1 = cat, 0 = not-cat
Y = np.array([1, 0, 1, 0])

# Seed for reproducibility
np.random.seed(42)
input_size = 4
hidden_size = 3
output_size = 1

# Hidden layer
weights1 = np.random.randn(input_size, hidden_size)
bias1 = np.zeros(hidden_size)
print(f"weights1: {weights1}")

# Output layer
weights2 = np.random.randn(hidden_size, output_size)
bias2 = np.zeros(output_size)
print(f"weights2: {weights2}")

# training

# Learning rate
lr = 0.1

# Training loop
for epoch in range(1000):
    total_loss = 0

    for x,y in zip(X,Y):
        ### --- Forward pass ---
        z1 = np.dot(x, weights1) + bias1
        a1 = relu(z1)
        z2 = np.dot(a1, weights2) + bias2
        pred = sigmoid(z2[0])

        ### --- Loss ---
        loss = binary_cross_entropy(pred, y)
        total_loss += loss

        ### --- Backpropagation ---

        # Output layer
        dL_dpred = -(y / (pred + 1e-8)) + ((1 - y) / (1 - pred + 1e-8))
        dpred_dz2 = pred * (1 - pred)
        dL_dz2 = dL_dpred * dpred_dz2

        dL_dw2 = a1[:, np.newaxis] * dL_dz2
        dL_db2 = dL_dz2

        # Hidden layer
        dz2_da1 = weights2.flatten()
        dL_da1 = dL_dz2 * dz2_da1
        drelu_dz1 = (z1 > 0).astype(float)
        dL_dz1 = dL_da1 * drelu_dz1

        dL_dw1 = x[:, np.newaxis] * dL_dz1
        dL_db1 = dL_dz1

        # Update all weights and biases
        weights2 -= lr * dL_dw2
        bias2 -= lr * dL_db2

        weights1 -= lr * dL_dw1
        bias1 -= lr * dL_db1

    # Print every 100 epochs
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss:.4f}")

# Step 1: Make Predictions After Training
print("\n--- Evaluation ---")
for x, y in zip(X, Y):
    # Forward pass
    z1 = np.dot(x, weights1) + bias1
    a1 = relu(z1)
    z2 = np.dot(a1, weights2) + bias2
    pred = sigmoid(z2[0])
    
    label = 1 if pred > 0.5 else 0
    print(f"Input: {x}, Predicted: {pred:.3f} → Label: {label} (True: {y})")

# Step 2: Optional — Track Accuracy
correct = 0
for x, y in zip(X, Y):
    z1 = np.dot(x, weights1) + bias1
    a1 = relu(z1)
    z2 = np.dot(a1, weights2) + bias2
    pred = sigmoid(z2[0])
    label = 1 if pred > 0.5 else 0
    correct += (label == y)

accuracy = correct / len(Y)

# Step 3: (Optional Bonus) Plot Loss Over Time
losses = []
for epoch in range(1000):
    total_loss = 0
    for x, y in zip(X, Y):
        ...
    losses.append(total_loss)

print(f"\nAccuracy: {accuracy * 100:.2f}%")

import matplotlib.pyplot as plt

plt.plot(losses)
plt.title("Loss over epochs")
plt.xlabel("Epoch")
plt.ylabel("Total Loss")
plt.grid(True)
plt.show()