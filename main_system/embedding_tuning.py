
import tensorflow as tf 
import numpy as np
import os.path as path
import pandas as pd 
from tensorflow.keras import layers
from tensorflow.keras.callbacks import Callback
from kerastuner import HyperModel
from kerastuner.tuners import RandomSearch
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from collections import Counter 

def initalize_tensorflow_gpu(memory_limit):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

def gloveEmbedding(d, word_index):
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

  embedding_matrix = np.zeros((len(word_index) + 1, d))
  for word, i in word_index.items():
      embedding_vector = embeddings_word_weight_dict.get(word)
      if embedding_vector is not None:
          embedding_matrix[i] = embedding_vector

  return embedding_matrix

def stoplist(file_name = "stopwords.txt"):
  stopwords_txt = open(path.join('preprocessing', file_name))
  stoplist = []
  for line in stopwords_txt:
      values = line.split()
      stoplist.append(values[0])
  stopwords_txt.close()
  return stoplist

def print_stats(test_loss, test_acc, test_precision, test_recall):
    F1 = 2 * (test_precision * test_recall) / (test_precision + test_recall)

    print('---------------')
    print('Test Loss: {}'.format(test_loss))
    print('Test Accuracy: {}'.format(test_acc))
    print('Test Precision: {}'.format(test_precision))
    print('Test Recall: {}'.format(test_recall))
    print('---------------')
    print('Test F1: {}'.format(F1))
    return F1


def load_data(n_classes, n_words, stop_words = True):
    
    train_df = pd.read_pickle(path.join('pandas_data', 'aspect_category_detection_train_'+str(n_classes)+'_classes.pkl'))
    test_df = pd.read_pickle(path.join('pandas_data','aspect_category_detection_test_'+str(n_classes)+'_classes.pkl'))

    if stop_words:
        stopwords = stoplist()
        train_df['text'] = train_df['text'].apply(lambda x: ' '.join([item for item in x.split() if item not in stopwords]))
        test_df['text'] = test_df['text'].apply(lambda x: ' '.join([item for item in x.split() if item not in stopwords]))

    tokenizer = tf.keras.preprocessing.text.Tokenizer(
        num_words=n_words,
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

    return x_train, y_train, x_val, y_val, x_test, y_test, word_index

def build_model(glove_embedding, trainable_embedding = False):

    glove_matrix = gloveEmbedding(glove_embedding, word_index)
    embedding_layer = layers.Embedding(len(word_index) + 1,
        glove_embedding,
        weights=[glove_matrix],
        trainable=trainable_embedding)

    model = tf.keras.Sequential()
    model.add(embedding_layer)

    model.add(tf.keras.layers.GlobalMaxPooling1D())
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.Dense(16, activation='sigmoid'))

    METRICS = [
        tf.keras.metrics.BinaryAccuracy(name='accuracy'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
    ]

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss='binary_crossentropy',
        metrics=[METRICS])
    return model

initalize_tensorflow_gpu(1024)

earlystop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)  
glove_embedding_array = [50, 100, 200, 300]

data_df = pd.read_pickle(path.join('main_system', path.join('aspect', 'aspect_embedding_layer.pkl')))

ea = glove_embedding_array 

for i in range(0, len(ea)):
    embedding_type = 'Trainable GloVe seed' 
    data_df = data_df.append({'type': embedding_type, 'dimension': int(ea[i]), 'f1': 0}, ignore_index=True)

    for k in range(1,6):
        x_train, y_train, x_val, y_val, x_test, y_test, word_index = load_data(16, 1750)

        model = build_model(ea[i], True)
        history = model.fit(x_train, 
            y_train, 
            epochs=250,
            validation_data=(x_val, y_val),
            callbacks=[earlystop_callback],
            verbose = 1)     


        test_loss, test_acc, test_precision, test_recall = model.evaluate(x_test, y_test)
        test_f1 = print_stats(test_loss, test_acc, test_precision, test_recall)
        
        if test_f1 > data_df.at[i,'f1']:
             data_df.at[i,'f1'] = test_f1

for i in range(len(ea), 2 * len(ea)):

    embedding_type = 'GloVe layer'
    data_df = data_df.append({'type': embedding_type, 'dimension': int(ea[i]), 'f1': 0}, ignore_index=True)

    for k in range(1,6):

        x_train, y_train, x_val, y_val, x_test, y_test, word_index = load_data(16, 1750)

        model = build_model(ea[i], False)
        history = model.fit(x_train, 
            y_train, 
            epochs=150,
            validation_data=(x_val, y_val),
            callbacks=[earlystop_callback],
            verbose = 1)     


        test_loss, test_acc, test_precision, test_recall = model.evaluate(x_test, y_test)
        test_f1 = print_stats(test_loss, test_acc, test_precision, test_recall)
        
        if test_f1 > data_df.at[i+len(ea),'f1']:
             data_df.at[i+len(ea),'f1'] = test_f1
             
print(data_df)
data_df.to_pickle(path.join('main_system', path.join('aspect', 'aspects_glove.pkl')))

# df = pd.read_pickle(path.join('main_system', path.join('aspect', 'aspect_baselinenn_data')))
# print(df)