import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.datasets import mnist

# ===================================
# Load and preprocess MNIST
# ===================================
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Flatten and normalize
x_train_flat = x_train.reshape(x_train.shape[0], -1).astype("float32") / 255.0
x_test_flat = x_test.reshape(x_test.shape[0], -1).astype("float32") / 255.0

# ===================================
# Define model
# ===================================
model = keras.Sequential([
    layers.Dense(256, activation='relu', input_shape=(784,)),
    layers.Dropout(0.4),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# ===================================
# Train
# ===================================
history = model.fit(
    x_train_flat, y_train,
    validation_split=0.2,
    epochs=12,
    batch_size=128,
    verbose=1
)

# ===================================
# Evaluate
# ===================================
loss, accuracy = model.evaluate(x_test_flat, y_test)
print(f"Test Accuracy: {accuracy*100:.2f}%")
print(f"Test Loss: {loss:.2f}")

# Simple Accuracy Plot
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Single Image Prediction (like your output)
idx = 0 
plt.imshow(x_test[idx].reshape(28, 28), cmap='gray')
plt.title(f"Actual: {y_test[idx]}")
plt.axis('off')
plt.show()

pred = np.argmax(model.predict(x_test_flat[idx:idx+1]))
print(f"Predicted Digit: {pred}")
