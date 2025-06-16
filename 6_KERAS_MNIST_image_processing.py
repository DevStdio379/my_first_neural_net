#  Step 1: Install TensorFlow and Other Dependencies
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

# Step 2: Load and Prepare the Dataset
# Load dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize pixel values from [0,255] to [0,1]
x_train = x_train / 255.0
x_test = x_test / 255.0

# Flatten 28x28 images into 784-length vectors
x_train = x_train.reshape(-1, 784)
x_test = x_test.reshape(-1, 784)

# One-hot encode the labels (e.g. 5 → [0 0 0 0 0 1 0 0 0 0])
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Step 3: Build the Neural Network
model = Sequential([
    Dense(128, activation='relu', input_shape=(784,)),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax') # 10-digits (0-9)
])

# Step 4: Compile the Model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Step 5: Train the Model
model.fit(x_train, y_train, epochs=5, batch_size=32, validation_split=0.1)

# Step 6: Evaluate the Model
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_acc:.2f}")

# Step 7: Predict and Visualize
import matplotlib.pyplot as plt

predictions = model.predict(x_test)

# Show 5 test images with predictions
for i in range(10):
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
    plt.title(f"Predicted: {np.argmax(predictions[i])}")
    plt.axis('off')
    plt.show()
