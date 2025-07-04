import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Load MNIST test data
mnist = tf.keras.datasets.mnist
(_, _), (x_test, y_test) = mnist.load_data()
x_test = x_test / 255.0

# Load the saved model
model = tf.keras.models.load_model("mnist_model.h5")

# Wrap with softmax to get probabilities
probability_model = tf.keras.Sequential([
    model,
    tf.keras.layers.Softmax()
])

# Make predictions on entire test set
predictions = probability_model.predict(x_test)

# Show incorrect predictions
for i in range(len(x_test)):
    predicted_label = np.argmax(predictions[i])
    actual_label = y_test[i]

    if predicted_label != actual_label:
        plt.imshow(x_test[i], cmap="gray")
        plt.title(f"Wrong prediction\nPredicted: {predicted_label}, Actual: {actual_label}")
        plt.axis("off")
        plt.show()
