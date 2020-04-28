
import tensorflow as tf 
import numpy as np
import os.path as path
import pandas as pd 
import matplotlib.pyplot as plt
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

def initalize_tensorflow_cpu():
    my_devices = tf.config.experimental.list_physical_devices(device_type='CPU')
    tf.config.experimental.set_visible_devices(devices= my_devices, device_type='CPU')




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


# define the model


def build_model(input_length, n_input_layers, kernel_array):

    glove_matrix = gloveEmbedding(300, word_index)
    embedding_layer = layers.Embedding(len(word_index) + 1,
        300,
        weights=[glove_matrix],
        trainable=True)

    inputs = []
    embeddings = []
    conv = []
    drop = []
    pool = []
    flat = []

    input_matrix = []

    for i in range(n_input_layers):
        inputs.append(layers.Input(shape=(input_length,)))
        embeddings.append(embedding_layer(inputs[i]))
        conv.append(layers.Conv1D(filters=32, kernel_size=kernel_array[i], activation='relu')(embeddings[i]))
        drop.append(layers.Dropout(0.5)(conv[i]))
        pool.append(layers.MaxPooling1D(pool_size=2)(drop[i]))
        flat.append(layers.Flatten()(pool[i]))
        input_matrix.append(flat[i])

    if n_input_layers == 1:
        merged_dense = tf.keras.layers.Dense(128, activation='relu')(input_matrix[0])
    else: 
        merged_inputs = tf.keras.layers.concatenate(input_matrix)
        merged_dense = tf.keras.layers.Dense(128, activation='relu')(merged_inputs)

    

    outputs = tf.keras.layers.Dense(16, activation='sigmoid')(merged_dense)

    model = tf.keras.Model(inputs=inputs, outputs = outputs)
    # model = tf.keras.Sequential()

    # model.add(embedding_layer)

    # # model.add(tf.keras.layers.GlobalMaxPooling1D())
    # # model.add(tf.keras.layers.LSTM(256, activation='relu'))

    # model.add(tf.keras.layers.Dense(256, activation='relu'))
    # model.add(tf.keras.layers.Dense(16, activation='sigmoid'))

	# model = Model(inputs=[inputs1, inputs2, inputs3], outputs=outputs)

    METRICS = [
        tf.keras.metrics.BinaryAccuracy(name='accuracy'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
    ]

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss='binary_crossentropy',
        metrics=[METRICS])

    # tf.keras.utils.plot_model(model, show_shapes=True, to_file='multichannel.png')

    return model

initalize_tensorflow_gpu(1024)

earlystop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)  
glove_embedding_array = [50, 100, 200, 300]

# data_df = pd.read_pickle(path.join('main_system', path.join('aspect', 'aspect_embedding_layer.pkl')))
data_df = pd.DataFrame(columns = ['type', 'dimension', 'f1'])


x_train, y_train, x_val, y_val, x_test, y_test, word_index = load_data(16, 1750)

# print(x_train[0])
input_length = x_train.shape[1]

chanels = 4

import itertools

# x_train = list(itertools.chain.from_iterable(itertools.repeat(x, chanels) for x in x_train))
# x_test = list(itertools.chain.from_iterable(itertools.repeat(x, chanels) for x in x_test))

# x_train = np.repeat(x_train,chanels)
# x_test = np.repeat(x_test,chanels)
x_train_arr = []
x_val_arr = []
x_test_arr = []
kernel_array = []

for i in range(0, chanels):
    x_train_arr.append(x_train)
    x_val_arr.append(x_val)
    x_test_arr.append(x_test)
    kernel_array.append(len(kernel_array)+3)

model = build_model(input_length, chanels, [2,4,6,8])
history = model.fit([x_train_arr, x_train], 
    y_train, 
    epochs=250,
    validation_data=([x_val_arr,x_val], y_val),
    callbacks=[earlystop_callback],
    verbose = 1)     
test_loss, test_acc, test_precision, test_recall = model.evaluate(x_test_arr, y_test)
test_f1 = print_stats(test_loss, test_acc, test_precision, test_recall)


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

