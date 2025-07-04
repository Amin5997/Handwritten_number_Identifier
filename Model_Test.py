import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Load MNIST test data
mnist = tf.keras.datasets.mnist
(_, _), (x_test, y_test) = mnist.load_data()
x_test = x_test / 255.0

# Load the saved model
model = tf.keras.models.load_model("mnist_model.h5")

# Make predictions
probability_model = tf.keras.Sequential([
    model,
    tf.keras.layers.Softmax()
])

predictions = probability_model.predict(x_test[:5])

# Print predictions
for i, pred in enumerate(predictions):
    print(f"Image {i}: Predicted = {np.argmax(pred)}, Actual = {y_test[i]}")

# Optional: Display one image
plt.imshow(x_test[0], cmap="gray")
plt.title(f"Predicted: {np.argmax(predictions[0])}, Actual: {y_test[0]}")
plt.show()
