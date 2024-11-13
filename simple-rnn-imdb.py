import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
max_features = 10000 
maxlen = 200 
batch_size = 32

 
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)


model = Sequential()
model.add(Embedding(max_features, 128))
model.add(SimpleRNN(128, activation='tanh')) 
model.add(Dropout(0.5)) 
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

history = model.fit(x_train, y_train, batch_size=batch_size, epochs=20, validation_split=0.2, callbacks=[early_stopping])

score, accuracy = model.evaluate(x_test, y_test, batch_size=batch_size)


print(f'Test score: {score:.4f}')
print(f'Test accuracy: {accuracy:.4f}')

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.show()


def decode_review(review):
  word_index = imdb.get_word_index()
  reverse_word_index = {value: key for key, value in word_index.items()}
  decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in review])
  return decoded_review

sample_reviews = [
"This movie was fantastic! I loved it.",
"I didn't like this film at all. It was boring and too long.",
"An average film, nothing special.",
]


def preprocess_reviews(reviews):
  encoded_reviews = []
  word_index = imdb.get_word_index()
  for review in reviews:
    encoded_review = [word_index.get(word.lower(), 0) + 3 for word in review.split()]
  encoded_reviews.append(encoded_review)
  return sequence.pad_sequences(encoded_reviews, maxlen=maxlen)

encoded_sample_reviews = preprocess_reviews(sample_reviews)
predictions = model.predict(encoded_sample_reviews)
predicted_classes = (predictions > 0.5).astype("int32") 

for review, prediction in zip(sample_reviews, predicted_classes):
  sentiment = "Positive" if prediction[0] == 1 else "Negative"
print(f"Review: {review}\nSentiment: {sentiment}\n")
