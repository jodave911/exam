import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.activations import relu
import numpy as np
import matplotlib.pyplot as plt
# Load and preprocess the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0 # Normalize the images
y_train, y_test = to_categorical(y_train), to_categorical(y_test) # Convert labels to categorical
# Define the model creation function
def create_model(hidden_units=None):
 model = models.Sequential([
 layers.Flatten(input_shape=(32, 32, 3)),
 layers.Dense(hidden_units[0], activation=relu),
 layers.Dense(hidden_units[1], activation=relu),
 layers.Dense(hidden_units[2], activation=relu),
 layers.Dense(10, activation='softmax') # Output layer
 ])
 return model
# Initialize results dictionary and counter
results_dict = {}
counter = 1
# Create, compile, and train the model
model = create_model(hidden_units=[512, 256, 128])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.2)
# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test)
model_info = f"Test accuracy: {round(test_acc * 100, 4)}%"
results_dict[counter] = model_info
counter += 1
# Print results
for key, value in results_dict.items():
 print(f"Run {key}: {value}")


import random
random_index = random.randint(0, len(test_images) - 1)
image = test_images[random_index]
label = test_labels[random_index][0]


for i in range(5):
  random_index = random.randint(0, len(test_images) - 1)
  image = test_images[random_index]
  label = test_labels[random_index].argmax()  # Get the index of the highest probability
  print(f"The class name of the random image is: {class_names[label]}")
  plt.imshow(image)
  plt.show()
