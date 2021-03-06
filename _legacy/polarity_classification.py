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




# # 50 Epochs
# # Test Loss: 0.5253028400084201
# # Test Accuracy: 0.730337083339691
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
# model.add(tf.keras.layers.Dense(128, activation='sigmoid'))

# model.add(tf.keras.layers.LSTM(lstm_output_size))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))


sgd = tf.keras.optimizers.SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True)
adam = tf.keras.optimizers.Adam(1e-4)

model.summary()

METRICS = [
      tf.keras.metrics.BinaryAccuracy(name='accuracy'),
      tf.keras.metrics.Precision(name='precision'),
      tf.keras.metrics.Recall(name='recall'),
]

model.compile(loss='categorical_hinge',
              optimizer=adam,
              metrics=METRICS)
              

# history = model.fit(x_train, 
#                     y_train, 
#                     epochs=75,
#                     validation_data=(x_val, y_val),
#                     verbose = 1
                    
#                     )   

# test_loss, test_acc, precision, recall = model.evaluate(x_test, y_test)

# F1 = 2 * (precision * recall) / (precision + recall)

# print('Test Loss: {}'.format(test_loss))
# print('Test Accuracy: {}'.format(test_acc))
# print(F1)

# model.save('polarity_classification_model') 

model = tf.keras.models.load_model('polarity_classification_model')

sentence = []
text = "I love the laptop, it is so fast! It is very expensive though."
text_processed = re.sub(r'[^\w\s]','',text.lower())


sentence.append(text_processed)
sentence_seq = tokenizer.texts_to_sequences(sentence)
sentence_vector = tf.keras.preprocessing.sequence.pad_sequences(sentence_seq, padding='post', maxlen=73)

print(model.predict([[sentence_vector[0]]])[0][0])




# category_dict = {'LAPTOP#GENERAL': 0, 'LAPTOP#OPERATION_PERFORMANCE': 1, 'LAPTOP#DESIGN_FEATURES': 2, 'LAPTOP#QUALITY': 3, 'LAPTOP#MISCELLANEOUS': 4, 'LAPTOP#USABILITY': 5, 'SUPPORT#QUALITY': 6, 'LAPTOP#PRICE': 7, 'COMPANY#GENERAL': 8, 'BATTERY#OPERATION_PERFORMANCE': 9, 'LAPTOP#CONNECTIVITY': 10, 'DISPLAY#QUALITY': 11, 'LAPTOP#PORTABILITY': 12, 'OS#GENERAL': 13, 'SOFTWARE#GENERAL': 14, 'KEYBOARD#DESIGN_FEATURES': 15}
# category_dict_inverted = {v: k for k, v in category_dict.items()}
# reverse_word_map = dict(map(reversed, tokenizer.word_index.items()))

# words = []
# for i in range(len(x_train[x])):
#   words.append(reverse_word_map[x_train[x][i]])
# print(words)

# for i in range(1,len(x_train)-1):
#       if model.predict([[x_train[i]]])[0][0] < 0.5:
#             print(i)#

