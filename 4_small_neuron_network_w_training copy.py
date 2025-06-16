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

# One-layer model: 1 neuron for binary classification
input_size = 4
weights = np.random.randn(input_size)
bias = 0.0

# training
print(f"initial_weights: {weights} | bias: {bias}")

# Learning rate
lr = 0.1

# Training loop
for epoch in range(1000):
    total_loss = 0

    for x,y in zip(X,Y):
        # Forward pass
        z = np.dot(weights, x) + bias
        pred = sigmoid(z)

        # Loss
        loss = binary_cross_entropy(pred, y)
        total_loss += loss

        # Backward pass (manual gradient) | chain rule # dLoss / dWeights
        dL_dpred = -(y / (pred + 1e-8)) + ((1 - y) / (1 - pred + 1e-8)) # dLoss / dPrediction
        dpred_dz = pred * (1 - pred) # derivative of sigmoid # dPrediction / dz
        dz_dw = x # dz / dWeights

        grad = dL_dpred * dpred_dz # apply chain rule dLoss / dz = (dLoss / dPrediction) * (dPrediction / dz)

        # Update weights and bias
        weights -= lr * grad * dz_dw
        bias -= lr * grad

    
    # Print every 100 epochs
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss:.4f}")


print("\nTrained weights:", weights)
print("Trained bias:", bias)

for x, y in zip(X, Y):
    pred = sigmoid(np.dot(weights, x) + bias)
    label = 1 if pred > 0.5 else 0
    print(f"Input: {x}, Predicted: {pred:.3f} → Label: {label} (True: {y})")