import tensorflow as tf
import pandas as pd 
from sklearn.feature_extraction.text import CountVectorizer
import string
import re
import numpy as np
from collections import Counter 
from sklearn.model_selection import train_test_split
import os.path as path


def gloveEmbedding(d):
  """
    Returns a d dimensional glove embedding 

    :param d: {50, 100, 200, 300} dimensional glove word embeddings
  """
  embeddings_word_weight_dict = {}
  file_name = "glove.6B."+str(d)+"d.txt"
  glove_txt = open(path.join('glove.6B', file_name))

  for line in glove_txt:
      values = line.split()
      word = values[0]
      weight = np.asarray(values[1:], dtype='float32')
      embeddings_word_weight_dict[word] = weight
  glove_txt.close()

  embedding_matrix = np.zeros((len(word_index) + 1, 100))
  for word, i in word_index.items():
      embedding_vector = embeddings_word_weight_dict.get(word)
      if embedding_vector is not None:
          embedding_matrix[i] = embedding_vector

  return embedding_matrix


def evaluateModel(m, x_test, y_test):
      """
        Prints out loss, accuracy, precision, recall and F-1 measure of a model

        :param m: tensorflow model 
      """
      test_loss, test_acc, test_precision, test_recall = model.evaluate(x_test, y_test)
      F1 = 2 * (test_precision * test_recall) / (test_precision + test_recall)

      print('---------------')
      print('Test Loss: {}'.format(test_loss))
      print('Test Accuracy: {}'.format(test_acc))
      print('Test Precision: {}'.format(test_precision))
      print('Test Recall: {}'.format(test_recall))
      print('---------------')
      print('Test F1: {}'.format(F1))



train_df = pd.read_pickle(path.join('pandas_data', 'aspect_category_detection_train.pkl'))
test_df = pd.read_pickle(path.join('pandas_data','aspect_category_detection_test.pkl'))



tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=1000,
                                                  oov_token="<unk>",
                                                  filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
tokenizer.fit_on_texts(train_df.text)


tokenizer.word_index['<pad>'] = 0
tokenizer.index_word[0] = '<pad>'

tokenizer.word_index['<unk>'] = 1
tokenizer.index_word[1] = '<unk>'


train_seqs = tokenizer.texts_to_sequences(train_df.text)
train_vector = tf.keras.preprocessing.sequence.pad_sequences(train_seqs, padding='post')
train_labels = np.stack(train_df.matrix, axis=0)

test_seqs = tokenizer.texts_to_sequences(test_df.text)

x_train, x_val, y_train, y_val = train_test_split(train_vector, train_labels, test_size = 0.2, random_state = 0)
x_test = tf.keras.preprocessing.sequence.pad_sequences(test_seqs, padding='post')
y_test = np.stack(test_df.matrix, axis=0)

word_index = tokenizer.word_index
vocab_size = len(Counter(" ".join(train_df.text).split(" ")))

# CNN
kernel_size = 5
filters = 64
pool_size = 4

glove_dimension = 100
glove_matrix = gloveEmbedding(glove_dimension)
embedding_layer = tf.keras.layers.Embedding(len(word_index) + 1,
                            glove_dimension,
                            weights=[glove_matrix],
                            trainable=False)




model = tf.keras.Sequential()
model.add(embedding_layer)

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

model.add(tf.keras.layers.Dense(1, activation='sigmoid'))




sgd = tf.keras.optimizers.SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True)
adam = tf.keras.optimizers.Adam(1e-4)

model.summary()

METRICS = [
      tf.keras.metrics.BinaryAccuracy(name='accuracy'),
      tf.keras.metrics.Precision(name='precision'),
      tf.keras.metrics.Recall(name='recall'),
]

model.compile(loss='mean_absolute_percentage_error',
              optimizer=adam,
              metrics=METRICS)
              

history = model.fit(x_train, 
                    y_train, 
                    epochs=15,
                    validation_data=(x_val, y_val),
                    verbose = 1
                    
                    )   


# Prints evaluation metrics of test data
evaluateModel(model, x_test, y_test)



"""
Accuracy: 
F1: 


"""


# model.save(path.join('tensorflow_models', 'a')) 


# F1 Measure 0.808

# model = tf.keras.models.load_model('polarity_classification_model')

# sentence = []
# text = "I love the laptop, it is so fast! It is very expensive though."
# text_processed = re.sub(r'[^\w\s]','',text.lower())


# sentence.append(text_processed)
# sentence_seq = tokenizer.texts_to_sequences(sentence)
# sentence_vector = tf.keras.preprocessing.sequence.pad_sequences(sentence_seq, padding='post', maxlen=73)

# print(model.predict([[sentence_vector[0]]])[0][0])
