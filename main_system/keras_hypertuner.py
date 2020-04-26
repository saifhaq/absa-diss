
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
        num_words=1250,
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
        trainable=False)

    model = tf.keras.Sequential()
    model.add(embedding_layer)

    model.add(layers.Dropout(rate=hp.Float(
        'dropout_1',
        min_value=0.0,
        max_value=0.2,
        default=0.1,
        step=0.025))
    )

    model.add(layers.Conv1D(
        filters=hp.Choice(
            'conv_filters',
            values=[16, 32, 64],
            default=32,
        ),
        kernel_size=3, 
        padding = 'same',
        activation='relu')
        )
    
    model.add(tf.keras.layers.MaxPooling1D(pool_size=2))

    model.add(layers.Dropout(rate=hp.Float(
        'dropout_2',
        min_value=0.0,
        max_value=0.2,
        default=0.1,
        step=0.025))
    )
    
    model.add(layers.Bidirectional(layers.LSTM(
        units=hp.Int('lstm_units',
        min_value=32,
        max_value=512,
        step=32),
        activation='relu')))

    model.add(layers.Dropout(rate=hp.Float(
        'dropout_3',
        min_value=0.0,
        max_value=0.2,
        default=0.1,
        step=0.025))
    )

    model.add(tf.keras.layers.Dense(8, activation='sigmoid'))

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
x_train, y_train, x_val, y_val, x_test, y_test, word_index = load_data(8)

tuner = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=50,
    executions_per_trial=3,
    directory='keras_tuner_dir',
    project_name='aspects')

tuner.search_space_summary()

earlystop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True) 

tuner.search(x_train, y_train,
             epochs=75,
             validation_data=(x_val, y_val),
             callbacks=[earlystop_callback])

models = tuner.get_best_models(num_models=2)
tuner.results_summary()

print(models[0])

test_loss, test_acc, test_precision, test_recall = models[0].evaluate(x_test, y_test)

F1 = 2 * (test_precision * test_recall) / (test_precision + test_recall)

print('---------------')
print('Test Loss: {}'.format(test_loss))
print('Test Accuracy: {}'.format(test_acc))
print('Test Precision: {}'.format(test_precision))
print('Test Recall: {}'.format(test_recall))
print('---------------')
print('Test F1: {}'.format(F1))

