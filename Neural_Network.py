import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 1. Load the MNIST dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 2. Normalize pixel values to [0, 1]
x_train, x_test = x_train / 255.0, x_test / 255.0

# 3. Build the neural network model
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),      # Flatten 28x28 images to 784 vector
    tf.keras.layers.Dense(128, activation='relu'),      # First hidden layer with 128 neurons
    tf.keras.layers.Dropout(0.2),                        # Dropout for regularization
    tf.keras.layers.Dense(64, activation='relu'),       # Second hidden layer with 64 neurons
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)                            # Output layer with 10 neurons (one per digit)
])

# 4. Compile the model
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

# 5. Train the model
model.fit(x_train, y_train, epochs=5)

# 6. Evaluate the model on the test set
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print('\nTest accuracy:', test_acc)

# 7. Add a Softmax layer to convert logits to probabilities
probability_model = tf.keras.Sequential([
    model,
    tf.keras.layers.Softmax()
])

# 8. Predict on the first 5 test images
predictions = probability_model.predict(x_test[:5])

for i, pred in enumerate(predictions):
    predicted_label = np.argmax(pred)
    print(f"Image {i}: Predicted label: {predicted_label}, Actual label: {y_test[i]}")

# Optional: visualize one test image and prediction
plt.imshow(x_test[0], cmap='gray')
plt.title(f"Predicted: {np.argmax(predictions[0])}, Actual: {y_test[0]}")
plt.show()

model.save("mnist_model.h5")