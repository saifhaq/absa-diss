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
        Takes *xml_path* and returns matrix corresponding to whether data is in *desired_category* or not, 
        with zeroes elswhere. 
        Ie: desired_category = 
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
            for opinion in opinions:
                if(opinion.attrib['category'] == desired_category):
                    label = 1
            
            sentences_list.append([sentence_id, sentence_text, label])

        except:
            pass

    return pd.DataFrame(sentences_list, columns = ["id", "text", "desired_category"])

def df_only_single_category(xml_path, category_dict, desired_category, n):
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
        category_matrix = np.zeros((n, ), dtype=int)
        sentence_text =  re.sub(r'[^\w\s]','',sentence_text.lower())

        try: 
            opinions = list(sentence)[1]
            found = False 

            for opinion in opinions:
                
                location = category_dict[opinion.attrib['category']]
                category_matrix[location] = 1

                if(opinion.attrib['category'] == desired_category):
                    found = True

            if found == True:
                sentences_list.append([sentence_id, sentence_text, category_matrix])

        except:
            pass

    return pd.DataFrame(sentences_list, columns = ["id", "text", "matrix"])

def df_predicted_category(xml_path, n, desired_category):
    """
        Generates dataframe 
    """
    tree = et.parse(xml_path)
    reviews = tree.getroot()
    sentences = reviews.findall('**/sentence')

    sentences_list = []
    
    for sentence in sentences:

        sentence_id = sentence.attrib['id']                
        sentence_text = sentence.find('text').text
        sentence_text =  re.sub(r'[^\w\s]','',sentence_text.lower())

        try: 
            matrix = np.zeros((n,), dtype=int)
            opinions = list(sentence)[1]
            for opinion in opinions:

                if(opinion.attrib['category'] == desired_category):      
                    sentences_list.append([sentence_id, sentence_text, matrix])

        except:
            pass

    return pd.DataFrame(sentences_list, columns = ["id", "text", "predicted_matrix"])


def df_something(xml_path, n, category_dict, empty_matrix_wanted = True):
    """
        Takes XML Training data and returns a pandas dataframe of sentences;
    """

    tree = et.parse(xml_path)
    reviews = tree.getroot()
    sentences = reviews.findall('**/sentence')

    sentences_list = []
    
    for sentence in sentences:

        sentence_id = sentence.attrib['id']                
        sentence_text = sentence.find('text').text
        category_matrix = np.zeros((n, ), dtype=int)
        sentence_text =  re.sub(r'[^\w\s]','',sentence_text.lower())

        try: 
            opinions = list(sentence)[1]
            categories = []
            for opinion in opinions:
            
                categories.append(opinion.attrib['category'])
                try:
                    location = category_dict[opinion.attrib['category']]
                    category_matrix[location] = 1
                except:
                    continue

                # category_matrix[i] = assigned(category_dict[opinion.attrib['category']])
            sentences_list.append([sentence_id, sentence_text, categories, category_matrix])

        except:
            pass

    return pd.DataFrame(sentences_list, columns = ["id", "text", "category", "matrix"])

def assign_category(xml_path, n):
    """
        Returns dictionary of n most common categories 
    """

    sentences = df_aspect_category(xml_path)
    categories = Counter(sentences.category).most_common(n)
    common_categories = [category_tuple[0] for category_tuple in categories]
    assigned = {}

    for i in range(0, len(common_categories)):
        assigned[common_categories[i]] = i 

    return assigned

TRAIN_XML_PATH = "ABSA16_Laptops_Train_SB1_v2.xml"
TEST_XML_PATH = "ABSA16_Laptops_Test_GOLD_SB1.xml"

n = 16

sentences = df_aspect_category(TRAIN_XML_PATH)
categories = Counter(sentences.category).most_common(n)

sentences_test = df_aspect_category(TRAIN_XML_PATH)
categories_test = Counter(sentences.category).most_common(n)

data_df = pd.DataFrame(columns = ["desired_category", "train_count"])


# print(categories)
DESIRED_CATEGORY = categories[0][0]

print(categories)
stoplist = stoplist()


category_dict = assign_category(TRAIN_XML_PATH, n)

pred_df = df_predicted_category(TEST_XML_PATH, n, DESIRED_CATEGORY)



actual_df = df_something(TEST_XML_PATH, n, category_dict, True)



predicted_matrix = pred_df.predicted_matrix
# actual_df = pd.read_pickle(path.join('pandas_data', 'aspect_category_detection_test_'+str(n)+'_classes.pkl'))
# actual_df = df_actual(TEST_XML_PATH, n, categories_test)
# category_dict = {'LAPTOP#GENERAL': 0, 'LAPTOP#OPERATION_PERFORMANCE': 1, 'LAPTOP#DESIGN_FEATURES': 2, 'LAPTOP#QUALITY': 3, 'LAPTOP#MISCELLANEOUS': 4, 'LAPTOP#USABILITY': 5, 'SUPPORT#QUALITY': 6, 'LAPTOP#PRICE': 7, 'COMPANY#GENERAL': 8, 'BATTERY#OPERATION_PERFORMANCE': 9, 'LAPTOP#CONNECTIVITY': 10, 'DISPLAY#QUALITY': 11, 'LAPTOP#PORTABILITY': 12, 'OS#GENERAL': 13, 'SOFTWARE#GENERAL': 14, 'KEYBOARD#DESIGN_FEATURES': 15}
category_dict = assign_category(TRAIN_XML_PATH, n)
actual_df = df_something(TEST_XML_PATH, n, category_dict, True)

# print(actual_df)
actual_matrix = actual_df.matrix


pred= np.reshape(predicted_matrix.values, (predicted_matrix.shape[0]))
# s = [a[0] for a in pred]

# print(actual_matrix[0])
#array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])]
# print(len(actual_matrix))
a  = []
p = []
for i in range(len(actual_matrix)):
    a.append(actual_matrix[i].tolist())

for i in range(len(predicted_matrix)):
    p.append(predicted_matrix[i].tolist())

# print(a)
# print(np.asarray(p).argmax(axis=1))
a = np.asarray(a)
p = np.asarray(p)

initalize_tensorflow_gpu(1024)



# model = tf.keras.models.load_model(path.join('acd', 'cnn_lstm_model'))
model = tf.keras.models.load_model(path.join('acd', 'dnn_model'))




test_only_single_matrix_df = df_only_single_category(TEST_XML_PATH, category_dict, DESIRED_CATEGORY, 16)
train_only_single_matrix_df = df_only_single_category(TRAIN_XML_PATH, category_dict, DESIRED_CATEGORY, 16)

tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=1750,
                                                  oov_token="<unk>",
                                                  filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
tokenizer.fit_on_texts(train_only_single_matrix_df.text)
tokenizer.word_index['<pad>'] = 0
tokenizer.index_word[0] = '<pad>'
tokenizer.word_index['<unk>'] = 1
tokenizer.index_word[1] = '<unk>'

train_seqs = tokenizer.texts_to_sequences(train_only_single_matrix_df.text)
train_vector = tf.keras.preprocessing.sequence.pad_sequences(train_seqs, padding='post')

test_seqs = tokenizer.texts_to_sequences(test_only_single_matrix_df.text)
x_test = tf.keras.preprocessing.sequence.pad_sequences(test_seqs, padding='post')

predicted = np.array(model.predict(x_test))
pred_labels = (predicted > 0.5).astype(np.int)

 

print(len(pred_labels))
print(len(test_only_single_matrix_df.matrix))
# print(predicted.argmax(axis=-1))

a  = []
p = []

for i in range(len(pred_labels)):
    a.append(pred_labels[i].tolist())

for i in range(len(test_only_single_matrix_df)):
    p.append(test_only_single_matrix_df.matrix[i].tolist())

a = np.asarray(a)
p = np.asarray(p)

# print(a)
# print(p)
print(confusion_matrix(a.argmax(axis=1), p.argmax(axis=1)))


# # print(confusion_matrix(a, p))

# # print('---------------')
# # print('Test Precision: {}'.format(precision_score(a, p, average="macro")))
# # print('Test Recall: {}'.format(recall_score(a, p, average="macro")))
# # print('Test Accuracy: {}'.format(accuracy_score(a, p)))
# # print('---------------')
# # print('Test F1: {}'.format(f1_score(a, p, average="macro")))


# # print(data_df)