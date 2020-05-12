
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
    Returns a *d* dimensional glove embedding matrix
    mapping each word index to it's glove weights

    :param d: {50, 100, 200, 300} dimensional glove word embeddings
    :param word_index: word to integer mapping created by tokenizer
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
    """
        Returns an array of stopwords, from each line of the 
        *file_name* text file
    """
    stopwords_txt = open(path.join('preprocessing', file_name))
    stoplist = []
    for line in stopwords_txt:
        values = line.split()
        stoplist.append(values[0])
    stopwords_txt.close()
    return stoplist

def print_stats(test_loss, test_acc, test_precision, test_recall, model_name):
    """
        Helper function using data from Tensorflow's model evaluation
        function to return the F1 and print model performance stats. 
        Also updates data_df df to contain model acc and f1
    """
    test_f1 = 2 * (test_precision * test_recall) / (test_precision + test_recall)

    data_df = pd.read_pickle(path.join('subjectivity', path.join('results', 'data_df.pkl')))

    try:
        best_acc = data_df[data_df['model']==model_name]['acc'].values[0]
    except: 
        best_acc = 0 

    if test_acc > best_acc:
        data_df = data_df[data_df.model != model_name]
        data_df = data_df.append({'model': model_name, 'acc': test_acc, 'f1': test_f1}, ignore_index=True)
        model.save(path.join('subjectivity', path.join('tf_models', model_name+"_model")))
        
    data_df.to_pickle(path.join('subjectivity', path.join('results', 'data_df.pkl')))

    print('---------------')
    print('Test Loss: {}'.format(test_loss))
    print('Test Accuracy: {}'.format(test_acc))
    print('Test Precision: {}'.format(test_precision))
    print('Test Recall: {}'.format(test_recall))
    print('---------------')
    print('Test F1: {}'.format(test_f1))
    return test_f1



def load_data(n_words, stop_words = True):

    train_df = pd.read_pickle(path.join('subjectivity', path.join('pandas_data', 'TRAIN_SUBJECTIVITY_ONE_HOT.pkl')))
    test_df = pd.read_pickle(path.join('subjectivity', path.join('pandas_data', 'TRAIN_SUBJECTIVITY_ONE_HOT.pkl')))

    if stop_words:
        stopwords = stoplist()
        train_df['text'] = train_df['text'].apply(lambda x: ' '.join([item for item in x.split() if item not in stopwords]))
        test_df['text'] = test_df['text'].apply(lambda x: ' '.join([item for item in x.split() if item not in stopwords]))

    tokenizer = tf.keras.preprocessing.text.Tokenizer(
        num_words=n_words,
        oov_token="<oov>",
        filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')

    tokenizer.fit_on_texts(train_df.text)

    tokenizer.word_index['<pad>'] = 0
    tokenizer.index_word[0] = '<pad>'

    tokenizer.word_index['<oov>'] = 1
    tokenizer.index_word[1] = '<oov>'

    train_seqs = tokenizer.texts_to_sequences(train_df.text)
    train_vector = tf.keras.preprocessing.sequence.pad_sequences(train_seqs, padding='post')
    train_labels = np.stack(train_df.subjectivity, axis=0)

    test_seqs = tokenizer.texts_to_sequences(test_df.text)

    x_train, x_val, y_train, y_val = train_test_split(train_vector, train_labels, test_size = 0.2, random_state = 0)

    x_test = tf.keras.preprocessing.sequence.pad_sequences(test_seqs, padding='post')
    y_test = np.stack(test_df.subjectivity, axis=0)

    word_index = tokenizer.word_index

    return x_train, y_train, x_val, y_val, x_test, y_test, word_index

def build_model(word_index):

    glove_matrix = gloveEmbedding(300, word_index)
    embedding_layer = layers.Embedding(len(word_index) + 1,
        300,
        weights=[glove_matrix],
        trainable=True)
    
    input_layer = layers.Input(shape=(input_length,))
    embedding = embedding_layer(input_layer)

    pooling = tf.keras.layers.GlobalMaxPooling1D()(embedding)
    dense = tf.keras.layers.Dense(256, activation='tanh')(pooling)

    outputs = tf.keras.layers.Dense(2, activation='softmax')(dense)
    model = tf.keras.Model(inputs=input_layer, outputs = outputs)


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

earlystop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=50, restore_best_weights=False)  
x_train, y_train, x_val, y_val, x_test, y_test, word_index = load_data(1750)

    
for k in range(0,1):
    input_length = x_train.shape[0]
    model = build_model(word_index)
    history = model.fit(x_train, 
        y_train, 
        epochs=250,
        validation_data=(x_val, y_val),
        callbacks=[earlystop_callback],
        verbose = 1)     
    test_loss, test_acc, test_precision, test_recall = model.evaluate(x_test, y_test)
    test_f1 = print_stats(test_loss, test_acc, test_precision, test_recall, 'dnn')
    

print(pd.read_pickle(path.join('subjectivity', path.join('results', 'data_df.pkl'))))