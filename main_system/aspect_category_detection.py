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
from nltk.corpus import stopwords
import nltk

def stoplist(file_name = "stopwords.txt"):
  stopwords_txt = open(path.join('preprocessing', file_name))
  stoplist = []
  for line in stopwords_txt:
      values = line.split()
      stoplist.append(values[0])
  stopwords_txt.close()
  return stoplist


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)



# train_df = pd.read_pickle(path.join('pandas_data', 'restaurants_aspect_category_detection_train.pkl'))
# test_df = pd.read_pickle(path.join('pandas_data','restaurants_aspect_category_detection_test.pkl'))

train_df = pd.read_pickle(path.join('pandas_data', 'aspect_category_detection_train.pkl'))
test_df = pd.read_pickle(path.join('pandas_data','aspect_category_detection_test.pkl'))

stoplist = stoplist()

train_df['text'] = train_df['text'].apply(lambda x: ' '.join([item for item in x.split() if item not in stoplist]))
test_df['text'] = test_df['text'].apply(lambda x: ' '.join([item for item in x.split() if item not in stoplist]))

# print(train_df.text)
# train_df.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)
# test_df.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)


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


glove_matrix = gloveEmbedding(100)
embedding_layer = tf.keras.layers.Embedding(len(word_index) + 1,
                            100,
                            weights=[glove_matrix],
                            trainable=False)



embedding_layer = tf.keras.layers.Embedding(len(word_index) + 1,
                            100,
                            weights=[glove_matrix],
                            trainable=False)



model = tf.keras.Sequential()
model.add(embedding_layer)





# model.add(tf.keras.layers.Conv1D(filters,
#                  kernel_size,
#                  padding='valid',
#                  activation='relu',
#                  strides=1))
# model.add(tf.keras.layers.MaxPooling1D(pool_size=pool_size))


# model.add(tf.keras.layers.Conv1D(filters,
#                  kernel_size,
#                  padding='valid',
#                  activation='relu',
#                  strides=1))
# model.add(tf.keras.layers.MaxPooling1D(pool_size=pool_size))



model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256)))

model.add(tf.keras.layers.Dense(8, activation='sigmoid'))



sgd = tf.keras.optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
adam = tf.keras.optimizers.Adam(1e-4)

model.summary()

METRICS = [
      tf.keras.metrics.BinaryAccuracy(name='accuracy'),
      tf.keras.metrics.Precision(name='precision'),
      tf.keras.metrics.Recall(name='recall'),
]

model.compile(loss='binary_crossentropy',
              optimizer=adam,
              metrics=METRICS)
              


history = model.fit(x_train, 
                    y_train, 
                    epochs=75,
                    validation_data=(x_val, y_val),
                    verbose = 1,                     
                    )   



test_loss, test_acc, test_precision, test_recall = model.evaluate(x_test, y_test)

F1 = 2 * (test_precision * test_recall) / (test_precision + test_recall)

print('---------------')
print('Test Loss: {}'.format(test_loss))
print('Test Accuracy: {}'.format(test_acc))
print('Test Precision: {}'.format(test_precision))
print('Test Recall: {}'.format(test_recall))
print('---------------')
print('Test F1: {}'.format(F1))



def plot_graphs(history, metric):
  plt.plot(history.history[metric])
  plt.plot(history.history['val_'+metric], '')
  plt.xlabel("Epochs")
  plt.ylabel(metric)
  plt.legend([metric, 'val_'+metric])
  plt.show()


plot_graphs(history, 'accuracy')
plot_graphs(history, 'precision')
plot_graphs(history, 'recall')

"""
GloVe, 128-layer bidirectional LSTM, category matrix length output with softmax & categorical_crossentropy & 50 epochs


Both
Test Loss: 0.5534250736236572
Test Accuracy: 0.8814352750778198
Test Precision: 0.4126637578010559
Test Recall: 0.5026595592498779
---------------
Test F1: 0.45323740529847023

Train excluding, test including  
2 Classes
  Acc : 0.6123244762420654
  F1 : 0.4132231273309003
4 Classes
  Acc : 0.8178626894950867
  F1 : 0.40204864896564246
8 Classes
  Acc : 0.9085413217544556
  F1 :  0.3495145487755204


Train included, test including 

 2 Classes
  Acc : 0.6123244762420654
  F1 : 0.4132231273309003
4 Classes
  Acc : 0.8568642735481262
  F1 : 0.3661485292893543
8 Classes
  Acc : 0.9102964401245117
  F1 : 0.35211269880547125   
 16 Classes
  Acc : 0.9481279253959656
  F1 : 0.30183727412615174   
"""
# Todo, check F1 and accuracy of top 4 classes when training on all data. 


# Test F1: 0.27393938058777995
# Test F1: 0.2880794737881986
# Test F1: 0.3110465132665278
# Test F1: 0.4668587686230232 4 classes

# Test F1: 0.85714 2 classes


### Containing empty matrix data: 
## 2 Classes of data
#     Test Accuracy: 0.8595944046974182
#     Test F1: 0.38144328169752045 
## 4 Classes of data
#     Test Accuracy: 0.8595944046974182
#     Test F1: 0.38144328169752045 



# Test F1: 0.41776 8 classes

# model.save(path.join('tensorflow_models', 'aspect_category_detection_model')) 

# model.save('aspect_category_model') 


# # {'LAPTOP#GENERAL': 0, 'LAPTOP#OPERATION_PERFORMANCE': 1, 'LAPTOP#DESIGN_FEATURES': 2, 'LAPTOP#QUALITY': 3, 'LAPTOP#MISCELLANEOUS': 4, 'LAPTOP#USABILITY': 5, 'SUPPORT#QUALITY': 6, 'LAPTOP#PRICE': 7}

# model = tf.keras.models.load_model(path.join('tensorflow_models', 'aspect_category_model')) 

# sentence = []
# text = "the price is very high"
# text_processed = re.sub(r'[^\w\s]','',text.lower())


# sentence.append(text_processed)
# sentence_seq = tokenizer.texts_to_sequences(sentence)
# sentence_vector = tf.keras.preprocessing.sequence.pad_sequences(sentence_seq, padding='post', maxlen=73)

# print(model.predict([[sentence_vector[0]]]))

"""
Train including test excluding
2 Classes
  Acc : 0.699999988079071
  F1 : 0.7044335198317166
4 Classes
  Acc : 
  F1 : 
8 Classes
  Acc : 
  F1 :   


Train excluding, test excluding 
2 Classes
  Acc : 0.824999988079071
  F1 : 0.8275861547538794
4 Classes
  Acc : 0.8305555582046509
  F1 : 0.6317907438379893
8 Classes
  Acc: 0.8781960010528564
  F1 : 0.41962773648035095

"""