import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.activations import relu
import numpy as np
import matplotlib.pyplot as plt
import random

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
y_train, y_test = to_categorical(y_train), to_categorical(y_test)


def create_model(hidden_units=None, initializer='glorot_uniform'):
  dropout_rate = 0.1
  model = models.Sequential([
      layers.Flatten(input_shape=(32, 32, 3)),
      layers.Dense(512, activation='relu'),
      layers.Dropout(dropout_rate),
      layers.Dense(256, activation='relu'),
      layers.Dropout(dropout_rate),
      layers.Dense(128, activation='relu'),
      layers.Dropout(dropout_rate),
      layers.Dense(64, activation='relu'),
      layers.Dropout(dropout_rate),
      layers.Dense(10, activation='softmax')
  ])
  return model


results_dict = {}
counter = 1

model_kaiming = create_model(hidden_units=[512, 256, 128], initializer='he_normal')
model_kaiming.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history_kaiming = model_kaiming.fit(x_train, y_train, epochs=20, batch_size=64, validation_split=0.2)

model_xavier = create_model(hidden_units=[512, 256, 128], initializer='glorot_uniform')
model_xavier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history_xavier = model_xavier.fit(x_train, y_train, epochs=20, batch_size=64, validation_split=0.2)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history_kaiming.history['accuracy'], label='Kaiming Train Accuracy')
plt.plot(history_xavier.history['accuracy'], label='Xavier Train Accuracy')
plt.title('Train Accuracy with Different Initializations')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history_kaiming.history['val_accuracy'], label='Kaiming Validation Accuracy')
plt.plot(history_xavier.history['val_accuracy'], label='Xavier Validation Accuracy')
plt.title('Validation Accuracy with Different Initializations')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
