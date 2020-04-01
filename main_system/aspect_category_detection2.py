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
import matplotlib.pyplot as plt
import os.path as path
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix

train_df = pd.read_pickle(path.join('pandas_data', 'aspect_category_detection_train.pkl'))
test_df = pd.read_pickle(path.join('pandas_data','aspect_category_detection_test.pkl'))


top_k = 5000
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=top_k,
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

# x, x_val, y, y_val = train_test_split(train_vector, train_labels, test_size = 0.2, random_state = 0)
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)


x_test = tf.keras.preprocessing.sequence.pad_sequences(test_seqs, padding='post')
y_test = np.stack(test_df.matrix, axis=0)

word_index = tokenizer.word_index
vocab_size = len(Counter(" ".join(train_df.text).split(" ")))


def gloveEmbedding(d):
  """
    Returns a d dimensional glove embedding 

    :param d: {50, 100, 200, 300} dimensional glove word embeddings
  """
  embeddings_word_weight_dict = {}
  file_name = "glove.6B."+str(d)+"d.txt"
  glove_txt = open(os.path.join('glove.6B', file_name))

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


# CNN
kernel_size = 5
filters = 64
pool_size = 4

# # {'LAPTOP#GENERAL': 0, 'LAPTOP#OPERATION_PERFORMANCE': 1, 'LAPTOP#DESIGN_FEATURES': 2, 'LAPTOP#QUALITY': 3, 'LAPTOP#MISCELLANEOUS': 4, 'LAPTOP#USABILITY': 5, 'SUPPORT#QUALITY': 6, 'LAPTOP#PRICE': 7}


model = tf.keras.models.load_model(path.join('tensorflow_models', 'aspect_category_detection_model')) 

sentence = []
text = "laptop battery is great but screen is trash"
text_processed = re.sub(r'[^\w\s]','',text.lower())


sentence.append(text_processed)
sentence_seq = tokenizer.texts_to_sequences(sentence)
sentence_vector = tf.keras.preprocessing.sequence.pad_sequences(sentence_seq, padding='post', maxlen=73)




predicted = model.predict([x_test])
y_actual = y_test
x = 10

y_pred = (predicted > 0.5)*1
print(len(y_pred))
print(len(y_actual))


# mean = np.mean(predicted == y_test)

# print(y_pred[0]*1)

# print('Test Mean: {}'.format(mean))
# print('---------------')
# print('Test Precision: {}'.format(precision_score(y_actual, y_pred, average="macro")))
# print('Test Recall: {}'.format(recall_score(y_actual, y_pred, average="macro")))
# print('---------------')
# print('Test F1: {}'.format(f1_score(y_actual, y_pred, average="macro")))

# # # print(model.predict([[sentence_vector[0]]]))
 