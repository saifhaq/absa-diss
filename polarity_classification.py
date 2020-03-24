import tensorflow as tf
import pandas as pd 
from sklearn.feature_extraction.text import CountVectorizer
import string
import re
import numpy as np
from collections import Counter 
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Dropout, Activation


df = pd.read_pickle('polarity.pkl')
test_df = pd.read_pickle('polarity_gold_test.pkl')
print(test_df)


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


labels = np.stack(df.polarity, axis=0)


test_seqs = tokenizer.texts_to_sequences(test_df.text)
x_test = tf.keras.preprocessing.sequence.pad_sequences(test_seqs, padding='post')
y_test = np.stack(test_df.polarity, axis=0)


x_train, x_val, y_train, y_val = train_test_split(cap_vector, labels, test_size = 0.1, random_state = 0)

# train_dataset = tf.data.Dataset.from_tensor_slices((X_train,y_train))
# test_dataset = tf.data.Dataset.from_tensor_slices((X_val,y_val))


# Convolution
kernel_size = 5
filters = 64
pool_size = 4

# LSTM
lstm_output_size = 70


# model = tf.keras.Sequential()
# model.add(tf.keras.layers.Embedding(vocab_size, 1))
# model.add(tf.keras.layers.GlobalAveragePooling1D())
# model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
# model.add(tf.keras.layers.Dense(1, activation=tf.nn.sigmoid))

# Reducing test size to 0.1 


# 50 Epochs
# Test Loss: 0.5253028400084201
# Test Accuracy: 0.730337083339691
# model = tf.keras.Sequential()
# model.add(tf.keras.layers.Embedding(vocab_size+2, 16))
# model.add(Dropout(0.25))
# model.add(tf.keras.layers.Conv1D(filters,
#                  kernel_size,
#                  padding='valid',
#                  activation='relu',
#                  strides=1))
# model.add(tf.keras.layers.MaxPooling1D(pool_size=pool_size))
# model.add(tf.keras.layers.LSTM(lstm_output_size))
# model.add(tf.keras.layers.Dense(1, activation=tf.nn.sigmoid))




# 50 Epochs
# Test Loss: 0.5253028400084201
# Test Accuracy: 0.730337083339691
model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(vocab_size+2, 16))
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
model.add(tf.keras.layers.LSTM(10))

# model.add(tf.keras.layers.LSTM(lstm_output_size))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))


sgd = tf.keras.optimizers.SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True)
adam = tf.keras.optimizers.Adam(1e-4)

model.summary()

model.compile(loss='categorical_hinge',
              optimizer=adam,
              metrics=['accuracy'])
              

history = model.fit(x_train, 
                    y_train, 
                    epochs=25,
                    validation_data=(x_val, y_val),
                    verbose = 1
                    
                    )   

test_loss, test_acc = model.evaluate(x_test, y_test)

print('Test Loss: {}'.format(test_loss))
print('Test Accuracy: {}'.format(test_acc))
# 