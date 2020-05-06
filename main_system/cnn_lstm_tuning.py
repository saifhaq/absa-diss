
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
from kerastuner import HyperModel
from kerastuner.tuners import RandomSearch

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


def build_model(hp):


    glove_matrix = gloveEmbedding(300, word_index)
    embedding_layer = layers.Embedding(len(word_index) + 1,
        300,
        weights=[glove_matrix],
        trainable=True)

    convs = []
    poolings = []

    input_layer = layers.Input(shape=(input_length,))
    embedding = embedding_layer(input_layer)
    bilstm = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(embedding)
    conc = tf.keras.layers.concatenate([embedding, bilstm])


    kernel_array = [1,2,3,4,5,6]

    n_channels = hp.Int(
                'n_channels',
                min_value=1,
                max_value=5,
                )

    filters = hp.Choice('filters',
        values=[64, 128, 256])




    if (n_channels == 1):
        conv = layers.Conv1D(filters=filters, kernel_size=kernel_array[0], activation='relu')(conc)
        channels_output = layers.GlobalMaxPooling1D()(conv) 

    else:
        for i in range(n_channels):
            conv1 = layers.Conv1D(filters=filters, kernel_size=kernel_array[i], activation='relu')(conc)

            convs.append(conv1)
            poolings.append(layers.GlobalMaxPooling1D()(convs[i]))

        channels_output = tf.keras.layers.concatenate(poolings)

    dropout = layers.Dropout(rate=hp.Float(
            'dropout1',
            min_value=0.0,
            max_value=0.5,
            default=0.25,
            step=0.05)
            )(channels_output)

    outputs = tf.keras.layers.Dense(16, activation='sigmoid')(dropout)
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


x_train, y_train, x_val, y_val, x_test, y_test, word_index = load_data(16, 1750)

input_length = x_train.shape[0]
# print(x_train.shape[1])
chanels = 3

x_train_arr = []
x_val_arr = []
x_test_arr = []
kernel_array = []

for i in range(0, chanels):
    x_train_arr.append(x_train)
    x_val_arr.append(x_val)
    x_test_arr.append(x_test)
    kernel_array.append(len(kernel_array)+1)


input_length = x_train.shape[1]


tuner = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=75,
    executions_per_trial=5,
    directory='cnn_lstm_randomsearch',
    project_name='model_tuning')

tuner.search_space_summary()

earlystop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=False) 

tuner.search(x_train, y_train,
             epochs=40,
             validation_data=(x_val, y_val),
             callbacks=[])

models = tuner.get_best_models(num_models=10)

data_df = pd.DataFrame(columns = ["n_channels", "dropout", "filters", "test_acc", "test_f1"])


for i in range(0, 10):
    models[i].save('cnn_lstm_model2_'+str(i))

    layer_names=[layer.name for layer in models[i].layers]
    matching = [s for s in layer_names if "conv1d" in s]
    n_channels = len(matching)
    dropout = models[i].layers[-2].get_config()['rate']
    filters = models[i].layers[4].get_config()['filters']
    test_loss, test_acc, test_precision, test_recall = models[i].evaluate(x_test, y_test)
    test_f1 = 2 * (test_precision * test_recall) / (test_precision + test_recall)

    data_df = data_df.append({'n_channels': n_channels, 'dropout': dropout, 'filters': filters, 'test_acc': test_acc, 'test_f1': test_f1}, ignore_index=True)

tuner.results_summary()
print(data_df)
data_df.to_pickle(path.join('acd', path.join('results', 'cnn_lstm_tuning_info.pkl')))


# model = tf.keras.models.load_model('cnn_lstm_model_'+str(4))
# model.summary()







# print('---------------')
# print('Test Loss: {}'.format(test_loss))
# print('Test Accuracy: {}'.format(test_acc))
# print('Test Precision: {}'.format(test_precision))
# print('Test Recall: {}'.format(test_recall))
# print('---------------')
# print('Test F1: {}'.format(F1))



# def plot_graphs(history, metric):
#   plt.plot(history.history[metric])
#   plt.plot(history.history['val_'+metric], '')
#   plt.xlabel("Epochs")
#   plt.ylabel(metric)
#   plt.legend([metric, 'val_'+metric])
#   plt.show()


# plot_graphs(history, 'accuracy')
# plot_graphs(history, 'precision')
# plot_graphs(history, 'recall')

