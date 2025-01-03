import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

import matplotlib.pyplot as plt

# Charger les données (exemple avec le jeu de données IMDB)
(train_texts,train_labels),(test_texts,test_labels)=keras.datasets.imdb.load_data(num_words=1000)

train_texts

word_index = tf.keras.datasets.imdb.get_word_index()

reverse_word_index =dict([(value,key) for ( key, value) in word_index.items()])

train_texts =[' '.join([reverse_word_index.get(i -3,'?') for i in seq ]) for seq in train_texts]

test_texts =[' '.join([reverse_word_index.get(i -3,'?') for i in seq ]) for seq in test_texts]

# Preprocess text data
max_length =256
tokenizer = Tokenizer(num_words=10000,oov_token='<OOV>')

tokenizer.fit_on_texts(train_texts)
train_sequences = tokenizer.texts_to_sequences(train_texts)
test_sequences = tokenizer.texts_to_sequences(test_texts)

train_padded = pad_sequences(train_sequences,maxlen=max_length,padding='post',truncating='post')
test_padded = pad_sequences(test_sequences,maxlen=max_length,padding='post',truncating='post')


embedding_dim = 64
model = Sequential([
    Embedding(input_dim=10000, output_dim=embedding_dim, input_length=max_length),
    LSTM(64, return_sequences=True),
    LSTM(64),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

epochs =2
batch_size =32
history=model.fit(train_padded, train_labels, epochs=epochs, batch_size=batch_size, validation_data=(test_padded, test_labels))


# Plot training history
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

predictions = model.predict(test_padded)

print(predictions[3])
print(test_labels[3])


# Evaluate the model on the test set
evaluation = model.evaluate(test_padded, test_labels)
print(f"Test Accuracy: {evaluation[1] *100:.2f}%")


# after improving its accuracy save the entire model to a file
model.save("LSTM.h5")