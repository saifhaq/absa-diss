import xml.etree.ElementTree as et
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn import svm
import numpy as np 
import os.path as path
import re
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix, roc_auc_score, multilabel_confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np 
import os.path as path 
import tensorflow as tf 


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

def stoplist(file_name = "stopwords.txt"):
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
        Also updates data_f1s df to contain model acc and f1
    """
    test_f1 = 2 * (test_precision * test_recall) / (test_precision + test_recall)

    data_f1s = pd.read_pickle(path.join('acd', path.join('results', 'data_f1s.pkl')))

    try:
        best_f1 = data_f1s[data_f1s['model']==model_name]['f1'].values[0]
    except: 
        best_f1 = 0 

    if test_f1 > best_f1:
        best_f1 = test_f1   
        data_f1s = data_f1s[data_f1s.model != model_name]
        data_f1s = data_f1s.append({'model': model_name, 'acc': test_acc, 'f1': test_f1}, ignore_index=True)
        model.save(path.join('acd', path.join('tf_models', model_name+"_model")))
        
    data_f1s.to_pickle(path.join('acd', path.join('results', 'data_f1s.pkl')))

    print('---------------')
    print('Test Loss: {}'.format(test_loss))
    print('Test Accuracy: {}'.format(test_acc))
    print('Test Precision: {}'.format(test_precision))
    print('Test Recall: {}'.format(test_recall))
    print('---------------')
    print('Test F1: {}'.format(test_f1))
    return test_f1

def load_data(n_classes, n_words, stop_words = True):

    train_df = pd.read_pickle(path.join('acd', path.join('pandas_data', 'MAIN_TRAIN_'+str(n_classes)+'_CLASSES.pkl')))
    test_df = pd.read_pickle(path.join('acd', path.join('pandas_data', 'MAIN_TEST_'+str(n_classes)+'_CLASSES.pkl')))

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

initalize_tensorflow_gpu(1024)
stopwords = stoplist()

TRAIN_XML_PATH = "ABSA16_Laptops_Train_SB1_v2.xml"
TEST_XML_PATH = "ABSA16_Laptops_Test_GOLD_SB1.xml"
n = 16


model_names = ['dnn', 'cnn', 'lstm', 'cnn_lstm']
for i in range (0, len(model_names)):
    data_df = pd.read_pickle(path.join('acd', path.join('results', 'data_df.pkl')))

    model_name = model_names[i]
    model = tf.keras.models.load_model(path.join('acd', path.join('tf_models', model_name+"_model")))

    x_train, y_train, x_val, y_val, x_test, y_test, word_index = load_data(n, 1750)
    input_length = x_train.shape[0]

    test_loss, test_acc, test_precision, test_recall = model.evaluate(x_test, y_test)
    test_f1 = print_stats(test_loss, test_acc, test_precision, test_recall, model_name)

print(pd.read_pickle(path.join('acd', path.join('results', 'data_f1s.pkl'))))