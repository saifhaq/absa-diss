
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

def load_data(n_classes):
    
    train_df = pd.read_pickle(path.join('pandas_data', 'aspect_category_detection_train_'+str(n_classes)+'_classes.pkl'))
    test_df = pd.read_pickle(path.join('pandas_data','aspect_category_detection_test_'+str(n_classes)+'_classes.pkl'))


    stopwords = stoplist()
    train_df['text'] = train_df['text'].apply(lambda x: ' '.join([item for item in x.split() if item not in stopwords]))
    test_df['text'] = test_df['text'].apply(lambda x: ' '.join([item for item in x.split() if item not in stopwords]))


    tokenizer = tf.keras.preprocessing.text.Tokenizer(
        num_words=1750,
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

    print(train_df.matrix[101])

    return x_train, y_train, x_val, y_val, x_test, y_test

x_train, y_train, x_val, y_val, x_test, y_test = load_data(16)

input_length = x_train.shape[1]

initalize_tensorflow_gpu(1024)

# model = tf.keras.models.load_model(path.join('acd', path.join('tf_models', 'cnn_lstm'+"_model")))

# data_df = data_df.append({'n_channels': n_channels, 'dropout': dropout, 'test_acc': test_acc, 'test_f1': test_f1}, ignore_index=True)

data_df = pd.read_pickle(path.join('acd', path.join('results', 'gem.pkl')))
print(data_df)

for i in range(0, 10):
    model = tf.keras.models.load_model('cnn_lstm'+"_model2_"+str(i))

    index = i-10
    layer_names=[layer.name for layer in model.layers]
    matching = [s for s in layer_names if "conv1d" in s]
    n_channels = len(matching)
    dropout = model.layers[-2].get_config()['rate']
    print(dropout)
    data_df['dropout'].iloc[index] = str('{0:.2f}'.format(dropout))
    print(data_df)
    # test_loss, test_acc, test_precision, test_recall = model.evaluate(x_test, y_test)
    # test_f1 = 2 * (test_precision * test_recall) / (test_precision + test_recall)
# print(n_channels)
# print(data_df)
lstm_neurons = model.layers[2].get_config()['layer']['config']['units']
data_df.to_pickle(path.join('acd', path.join('results', 'gem.pkl')))

# print(model.layers[4].get_config()['filters'])
# print(x_train)
# test_loss, test_acc, test_precision, test_recall = models[0].evaluate(x_test, y_test)

# F1 = 2 * (test_precision * test_recall) / (test_precision + test_recall)

# print('---------------')
# print('Test Loss: {}'.format(test_loss))
# print('Test Accuracy: {}'.format(test_acc))
# print('Test Precision: {}'.format(test_precision))
# print('Test Recall: {}'.format(test_recall))
# print('---------------')
# print('Test F1: {}'.format(F1))

