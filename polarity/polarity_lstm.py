
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

    # data_df = pd.read_pickle(path.join('polarity', path.join('results', 'data_df.pkl')))

    # try:
    #     best_f1 = data_df[data_df['model']==model_name]['f1'].values[0]
    # except: 
    #     best_f1 = 0 

    # if test_f1 > best_f1:
    #     best_f1 = test_f1   
    #     data_df = data_df[data_df.model != model_name]
    #     data_df = data_df.append({'model': model_name, 'acc': test_acc, 'f1': test_f1}, ignore_index=True)
    #     model.save(path.join('polarity', path.join('tf_models', model_name+"_model")))
    #     print("yes")
        
    # data_df.to_pickle(path.join('polarity', path.join('results', 'data_df.pkl')))

    print('---------------')
    print('Test Loss: {}'.format(test_loss))
    print('Test Accuracy: {}'.format(test_acc))
    print('Test Precision: {}'.format(test_precision))
    print('Test Recall: {}'.format(test_recall))
    print('---------------')
    print('Test F1: {}'.format(test_f1))
    return test_f1


def load_data(n_classes, n_words, stop_words = True):

    train_df = pd.read_pickle(path.join('polarity', path.join('pandas_data', 'TRAIN_POLARITY.pkl')))
    test_df = pd.read_pickle(path.join('polarity', path.join('pandas_data', 'TEST_POLARITY.pkl')))

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
    train_categories = np.stack(train_df.category, axis=0)
    train_labels = np.stack(train_df.polarity, axis=0)

    x1_train, x1_val, x2_train, x2_val, y_train, y_val = train_test_split(train_vector, train_categories, train_labels, test_size = 0.2, random_state = 0)

    test_seqs = tokenizer.texts_to_sequences(test_df.text)
    x1_test = tf.keras.preprocessing.sequence.pad_sequences(test_seqs, padding='post')
    x2_test = np.stack(test_df.category, axis=0)
    y_test = np.stack(test_df.polarity, axis=0)

    word_index = tokenizer.word_index

    return [x1_train, x2_train], y_train, [x1_val, x2_val], y_val, [x1_test, x2_test], y_test, word_index

def build_model(word_index, filters, kernel_array):
    
    input_layer = layers.Input(shape=(29,))
    intput_layer_category = layers.Input(shape=(16,))


    n_channels = len(kernel_array)
    glove_matrix = gloveEmbedding(300, word_index)
    embedding_layer = layers.Embedding(len(word_index) + 1,
        300,
        weights=[glove_matrix],
        trainable=True)
    
    convs = []
    poolings = []
    

    embedding = embedding_layer(input_layer)
    bilstm = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(embedding)
    conc = tf.keras.layers.concatenate([embedding, bilstm])

    if (n_channels == 1):
        conv = layers.Conv1D(filters=filters, kernel_size=kernel_array[0], activation='relu')(conc)
        channels_output = layers.GlobalMaxPooling1D()(conv) 

    else:
        for i in range(n_channels):
            conv1 = layers.Conv1D(filters=filters, kernel_size=kernel_array[i], activation='relu')(conc)

            convs.append(conv1)
            poolings.append(layers.GlobalMaxPooling1D()(convs[i]))

        channels_output = tf.keras.layers.concatenate(poolings)

    dropout = layers.Dropout(0.3)(channels_output)
    d2 = tf.keras.layers.Dense(16, activation='sigmoid')(dropout)




    category_dense = layers.Dense(16, activation='sigmoid')(intput_layer_category)
    merged_inputs = layers.concatenate([d2, category_dense])

    dense2 = layers.Dense(128)(merged_inputs)

    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(dense2)


    model = tf.keras.Model(inputs=[input_layer, intput_layer_category], outputs = outputs)


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

earlystop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=False)  

    
x_trains, y_train, x_vals, y_val, x_tests, y_test, word_index = load_data(16, 1750)
input_length = x_trains[0].shape[0]


model = build_model(word_index, 256, [1,2,3])
print(model.summary())
history = model.fit(x_trains, 
    y_train, 
    epochs=250,
    validation_data=(x_vals, y_val),
    callbacks=[earlystop_callback],
    verbose = 1)     

test_loss, test_acc, test_precision, test_recall = model.evaluate(x_tests, y_test)
test_f1 = print_stats(test_loss, test_acc, test_precision, test_recall, 'lstm')
    

# print(pd.read_pickle(path.join('polarity', path.join('results', 'data_df.pkl'))))