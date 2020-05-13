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

def assign_category(xml_path, n):
    """
        Returns dictionary of the *n* most common as the keys 
        The values of each key is the index of the category  
    """

    sentences = df_aspect_category(xml_path)
    categories = Counter(sentences.category).most_common(n)
    common_categories = [category_tuple[0] for category_tuple in categories]
    assigned = {}

    for i in range(0, len(common_categories)):
        assigned[common_categories[i]] = i 

    return assigned

initalize_tensorflow_gpu(1024)
stopwords = stoplist()

TRAIN_XML_PATH = "ABSA16_Laptops_Train_SB1_v2.xml"
TEST_XML_PATH = "ABSA16_Laptops_Test_GOLD_SB1.xml"
n = 16

sentences = df_aspect_category(TRAIN_XML_PATH)
categories = Counter(sentences.category).most_common(n)

data_df = pd.read_pickle(path.join('acd', path.join('results', 'data_df.pkl')))

model_name = 'lstm_tuned_model'

model = tf.keras.models.load_model(path.join('acd', path.join('tf_models', model_name)))


category_dict = assign_category(TRAIN_XML_PATH, n)
experiments_df = pd.DataFrame(columns = ["model", "acc", "f1"])
sentences = ['It is amazing and I love the screen hinge']

for i in range(0, len(sentences)):
    experiments_df = experiments_df.append({'text': sentences[i]}, ignore_index=True)



train_df = pd.read_pickle(path.join('acd', path.join('pandas_data', 'MAIN_TRAIN_'+str(n)+'_CLASSES.pkl')))
experiments_df['text'] = experiments_df['text'].apply(lambda x: ' '.join([item for item in x.split() if item not in stopwords]))
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=1750,
                                                        oov_token="<oov>",
                                                        filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')

tokenizer.fit_on_texts(train_df.text)
tokenizer.word_index['<pad>'] = 0
tokenizer.index_word[0] = '<pad>'
tokenizer.word_index['<oov>'] = 1
tokenizer.index_word[1] = '<oov>'


test_seqs = tokenizer.texts_to_sequences(experiments_df.text)
x_test = tf.keras.preprocessing.sequence.pad_sequences(test_seqs, padding='post')

predicted = np.array(model.predict(x_test))
pred_labels = (predicted > 0.5).astype(np.int)

# print(pred_labels)

data_df = pd.read_pickle(path.join('acd', path.join('pandas_data', 'BASELINE_TRAIN_LAPTOP#MISCELLANEOUS.pkl')))
data_df = pd.read_pickle(path.join('acd', path.join('pandas_data', 'BASELINE_TEST_LAPTOP#GENERAL.pkl')))

data_misc = data_df.loc[data_df['desired_category'] == 1]


mean_len = data_misc.text.str.len().groupby(data_misc.id).mean()
misc_texts = data_misc.text.to_list()

lens = []
for i in range(0, len(misc_texts)):
    arr = misc_texts[i].split(" ")
    print(arr)
    lens.append(len(arr))

print(sum(lens)/len(lens))
# print(data_misc)
#         total_test_samples = len(test_only_single_matrix_df)
#         TP = 0 
#         for k in range(total_test_samples):
#             if pred_labels[k][desired_category_index] == 1:
#                 TP +=1

#         acc = TP/total_test_samples

#         data_df.at[j, model_names[i]] = str('{0:.2f}'.format(acc*100)) + "%"


# data_df.to_pickle(path.join('acd', path.join('results', 'data_df.pkl')))

# print(data_df)