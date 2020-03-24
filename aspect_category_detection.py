import tensorflow as tf
import pandas as pd 
from sklearn.feature_extraction.text import CountVectorizer
import string
import re
import numpy as np
from collections import Counter 
from sklearn.model_selection import train_test_split
import os
from sklearn.metrics import f1_score

df = pd.read_pickle('aspect_category_detection_train_10_classes.pkl')
test_df = pd.read_pickle('aspect_category_detection_test_10_classes.pkl')



top_k = 5000
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=top_k,
                                                  oov_token="<unk>",
                                                  filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
tokenizer.fit_on_texts(df.text)


tokenizer.word_index['<pad>'] = 0
tokenizer.index_word[0] = '<pad>'

tokenizer.word_index['<unk>'] = 1
tokenizer.index_word[1] = '<unk>'


train_seqs = tokenizer.texts_to_sequences(df.text)
cap_vector = tf.keras.preprocessing.sequence.pad_sequences(train_seqs, padding='post')

vocab_size = len(Counter(" ".join(df.text).split(" ")))


labels = np.stack(df.matrix, axis=0)

x_train, x_val, y_train, y_val = train_test_split(cap_vector, labels, test_size = 0.2, random_state = 0)

test_seqs = tokenizer.texts_to_sequences(test_df.text)
x_test = tf.keras.preprocessing.sequence.pad_sequences(test_seqs, padding='post')
y_test = np.stack(test_df.matrix, axis=0)


word_index = tokenizer.word_index

embeddings_index = {}
f = open(os.path.join('glove.6B', 'glove.6B.100d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

embedding_matrix = np.zeros((len(word_index) + 1, 100))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector



# Convolution
kernel_size = 5
filters = 64
pool_size = 4

# LSTM
lstm_output_size = 70



embedding_layer = tf.keras.layers.Embedding(len(word_index) + 1,
                            100,
                            weights=[embedding_matrix],
                            trainable=False)

# 50 Epochs
# Test Loss: 0.5253028400084201
# Test Accuracy: 0.730337083339691
model = tf.keras.Sequential()
# model.add(tf.keras.layers.Embedding(vocab_size+1, 16))
model.add(embedding_layer)

# model.add(Dropout(0.2))

model.add(tf.keras.layers.Conv1D(filters,
                 kernel_size,
                 padding='valid',
                 activation='relu',
                 strides=1))
model.add(tf.keras.layers.MaxPooling1D(pool_size=pool_size))



model.add(tf.keras.layers.Conv1D(filters,
                 kernel_size,
                 padding='valid',
                 activation='relu',
                 strides=1))
model.add(tf.keras.layers.MaxPooling1D(pool_size=pool_size))
model.add(tf.keras.layers.LSTM(128))

# model.add(tf.keras.layers.LSTM(lstm_output_size))
model.add(tf.keras.layers.Dense(10, activation='sigmoid'))


sgd = tf.keras.optimizers.SGD(lr=0.0005, decay=1e-6, momentum=0.9, nesterov=True)
adam = tf.keras.optimizers.Adam(1e-4)

model.summary()

METRICS = [
      tf.keras.metrics.BinaryAccuracy(name='accuracy'),
      tf.keras.metrics.Precision(name='precision'),
      tf.keras.metrics.Recall(name='recall'),
]

model.compile(loss='binary_crossentropy',
              optimizer=sgd,
              metrics=METRICS)
              


history = model.fit(x_train, 
                    y_train, 
                    epochs=1000,
                    validation_data=(x_val, y_val),
                    verbose = 1,                     
                    )   



test_loss, test_acc, precision, recall = model.evaluate(x_test, y_test)

F1 = 2 * (precision * recall) / (precision + recall)

print('Test Loss: {}'.format(test_loss))
print('Test Accuracy: {}'.format(test_acc))
print(F1)
# print(history.history)
# 