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
import xml.etree.ElementTree as et
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix, roc_auc_score

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


train_df = pd.read_pickle(path.join('pandas_data', 'aspect_category_detection_train_8_classes.pkl'))
test_df = pd.read_pickle(path.join('pandas_data','aspect_category_detection_test_8_classes.pkl'))

# train_df = pd.read_pickle(path.join('pandas_data', 'aspect_category_detection_train.pkl'))
# test_df = pd.read_pickle(path.join('pandas_data','aspect_category_detection_test.pkl'))

stoplist = stoplist()

# train_df['text'] = train_df['text'].apply(lambda x: ' '.join([item for item in x.split() if item not in stoplist]))
# test_df['text'] = test_df['text'].apply(lambda x: ' '.join([item for item in x.split() if item not in stoplist]))

# print(train_df.text)
# train_df.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)
# test_df.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)


def df_aspect_category(xml_path):
    """
        Takes *xml_path* and returns dataframe of each sentence and corresponding category. 
        If sentence has multiple categories, the sentence is returned multiple times. 
        
        Dataframe returned as: [id, text, category, polarity] 
    """

    tree = et.parse(xml_path)
    reviews = tree.getroot()
    sentences = reviews.findall('**/sentence')

    sentences_list = []

    for sentence in sentences:

        sentence_id = sentence.attrib['id']                
        sentence_text = sentence.find('text').text

        try: 
            opinions = list(sentence)[1]

            for opinion in opinions:
                category = opinion.attrib['category']
                polarity = opinion.attrib['polarity']
                sentences_list.append([sentence_id, sentence_text, category, polarity])

        except:
            pass

    return pd.DataFrame(sentences_list, columns = ["id", "text", "category", "polarity"])


def df_single_category(xml_path, desired_category):
    """
        Takes *xml_path* and returns labels of data corresponding to whether data is in *desired_category* or not

    """

    tree = et.parse(xml_path)
    reviews = tree.getroot()
    sentences = reviews.findall('**/sentence')

    sentences_list = []
    
    for sentence in sentences:

        sentence_id = sentence.attrib['id']                
        sentence_text = sentence.find('text').text
        label = [0 , 1]
        sentence_text =  re.sub(r'[^\w\s]','',sentence_text.lower())

        try: 
            opinions = list(sentence)[1]
            for opinion in opinions:
                if(opinion.attrib['category'] == desired_category):
                    label = [1, 0]
            
            sentences_list.append([sentence_id, sentence_text, label])

        except:
            pass

    return pd.DataFrame(sentences_list, columns = ["id", "text", "desired_category"])

def df_predicted_category(xml_path, n):
    """
        Takes *xml_path* and returns labels of data corresponding to whether data is in *desired_category* or not

    """

    tree = et.parse(xml_path)
    reviews = tree.getroot()
    sentences = reviews.findall('**/sentence')

    sentences_list = []
    
    for sentence in sentences:

        sentence_id = sentence.attrib['id']                
        sentence_text = sentence.find('text').text
        label = 0
        sentence_text =  re.sub(r'[^\w\s]','',sentence_text.lower())

        try: 
            opinions = list(sentence)[1]
            matrix = np.zeros((n,), dtype=int)
            sentences_list.append([sentence_id, sentence_text, matrix])

        except:
            pass

    return pd.DataFrame(sentences_list, columns = ["id", "text", "predicted_matrix"])

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

n = 8

TRAIN_XML_PATH = "ABSA16_Laptops_Train_SB1_v2.xml"
TEST_XML_PATH = "ABSA16_Laptops_Test_GOLD_SB1.xml"
pred_df = df_predicted_category(TEST_XML_PATH, n)
sentences = df_aspect_category(TRAIN_XML_PATH)
categories = Counter(sentences.category).most_common(n)

sentences_test = df_aspect_category(TRAIN_XML_PATH)
categories_test = Counter(sentences.category).most_common(n)

for i in range(0,n):
    
    DESIRED_CATEGORY = categories[i][0]
    TRAIN_COUNT = categories[i][1]

    train_df = df_single_category(TRAIN_XML_PATH, DESIRED_CATEGORY)
    test_df = df_single_category(TEST_XML_PATH, DESIRED_CATEGORY)

    x_train, y_train = train_df.text, train_df.desired_category
    x_test, y_test = test_df.text, test_df.desired_category

    top_k = 1750
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
    train_labels = np.stack(train_df.desired_category, axis=0)

    test_seqs = tokenizer.texts_to_sequences(test_df.text)

    x_train, x_val, y_train, y_val = train_test_split(train_vector, train_labels, test_size = 0.2, random_state = 0)


    x_test = tf.keras.preprocessing.sequence.pad_sequences(test_seqs, padding='post')
    y_test = np.stack(test_df.desired_category, axis=0)

    word_index = tokenizer.word_index
    vocab_size = len(Counter(" ".join(train_df.text).split(" ")))

  # 00000000
    glove_matrix = gloveEmbedding(100)
    embedding_layer = tf.keras.layers.Embedding(len(word_index) + 1,
                                100,
                                weights=[glove_matrix],
                                trainable=False)

    model = tf.keras.Sequential()
    model.add(embedding_layer)
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128)))
    model.add(tf.keras.layers.Dense(2, activation='softmax'))

    adam = tf.keras.optimizers.Adam(1e-4)

    METRICS = [
          tf.keras.metrics.BinaryAccuracy(name='accuracy'),
          tf.keras.metrics.Precision(name='precision'),
          tf.keras.metrics.Recall(name='recall'),
    ]

    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=METRICS)
                  
    history = model.fit(x_train, 
                        y_train, 
                        epochs=75,
                        validation_data=(x_val, y_val),
                        verbose = 1,                     
                        )   


    predicted = model.predict(x_test)

    for j in range(0, len(predicted)):
        matrix = pred_df['predicted_matrix'][j] 
        if predicted[j][0] > predicted[j][1]:
          matrix[i] = 1 
        else:
          matrix[i] = 0


predicted_matrix = pred_df.predicted_matrix
actual_df = pd.read_pickle(path.join('pandas_data', 'aspect_category_detection_test_8_classes.pkl'))
print(actual_df)
actual_matrix = actual_df.matrix

a  = []
p = []
for i in range(len(actual_matrix)):
    a.append(actual_matrix[i].tolist())

for i in range(len(predicted_matrix)):
    p.append(predicted_matrix[i].tolist())


print('---------------')
print('Test Precision: {}'.format(precision_score(a, p, average="macro")))
print('Test Recall: {}'.format(recall_score(a, p, average="macro")))
print('Test Accuracy: {}'.format(accuracy_score(a, p)))
print('---------------')
print('Test F1: {}'.format(f1_score(a, p, average="macro")))

