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
from sklearn.svm import SVC

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix, roc_auc_score, multilabel_confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np 
import os.path as path 


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
        Takes *xml_path* and returns labels of data corresponding to whether data is in *desired_category* or not

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
            opinions = list(sentence)[1]
            for opinion in opinions:
                if(opinion.attrib['category'] == desired_category):

                    if(polarity == "positive"):
                        sentences_list.append([sentence_id, sentence_text, 1])

                    elif(polarity == "negative"):
                        sentences_list.append([sentence_id, sentence_text, 0])

        except:
            pass

    return pd.DataFrame(sentences_list, columns = ["id", "text", "desired_category"])


def df_actual(xml_path, n, category_dict):
    """
        Takes XML Training data and returns a pandas dataframe of sentences;
        returns duplicate sentences if each sentence has multiple aspects of polarity   
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
            opinions = list(sentence)[1]
            for opinion in opinions:
                category_matrix = np.zeros((16, ), dtype=int)

                try:
                    polarity = opinion.attrib['polarity']
                    
                    location = category_dict[opinion.attrib['category']]
                    category_matrix[location] = 1
    
                    if(polarity == "positive"):
                        sentences_list.append([sentence_id, sentence_text, category_matrix, 1])

                    elif(polarity == "negative"):
                        sentences_list.append([sentence_id, sentence_text, category_matrix, 0])

                except:
                    continue

        except:
            pass


    return pd.DataFrame(sentences_list, columns = ["id", "text", "category", "polarity"])

def df_predicted(xml_path, n, category_dict):
    """
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
            opinions = list(sentence)[1]
            for opinion in opinions:
                category_matrix = np.zeros((16, ), dtype=int)

                try:
                    polarity = opinion.attrib['polarity']
                    
                    location = category_dict[opinion.attrib['category']]
                    category_matrix[location] = 1
    
                    if(polarity == "positive"):
                        sentences_list.append([sentence_id, sentence_text, category_matrix, -2])

                    elif(polarity == "negative"):
                        sentences_list.append([sentence_id, sentence_text, category_matrix, -2])

                except:
                    continue

        except:
            pass


    return pd.DataFrame(sentences_list, columns = ["id", "text", "category", "polarity"])


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

TRAIN_XML_PATH = "ABSA16_Laptops_Train_SB1_v2.xml"
TEST_XML_PATH = "ABSA16_Laptops_Test_GOLD_SB1.xml"

n = 16

sentences = df_aspect_category(TRAIN_XML_PATH)
categories = Counter(sentences.category).most_common(n)

data_df = pd.read_pickle(path.join('polarity', path.join('results', 'data_df.pkl')))

category_dict = assign_category(TRAIN_XML_PATH, 16)
pred_df = df_predicted(TEST_XML_PATH, n, category_dict)


stoplist = stoplist()
for i in range(0,n):
    
    DESIRED_CATEGORY = categories[i][0]
    TRAIN_COUNT = categories[i][1]

    train_df = df_single_category(TRAIN_XML_PATH, DESIRED_CATEGORY)
    test_df = df_single_category(TEST_XML_PATH, DESIRED_CATEGORY)


    train_df['text'] = train_df['text'].apply(lambda x: ' '.join([item for item in x.split() if item not in stoplist]))
    test_df['text'] = test_df['text'].apply(lambda x: ' '.join([item for item in x.split() if item not in stoplist]))

    train_df_name = 'BASELINE_'+'TRAIN_'+DESIRED_CATEGORY + '.pkl'
    test_df_name =  'BASELINE_'+'TEST_'+DESIRED_CATEGORY + '.pkl'

    train_df.to_pickle(path.join('polarity', path.join('pandas_data', train_df_name)))
    test_df.to_pickle(path.join('polarity', path.join('pandas_data', test_df_name)))


    x_train, y_train = train_df.text, train_df.polarity
    x_test, y_test = test_df.text, test_df.polarity

    text_clf = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('svc', SGDClassifier()),    
        ])

    text_clf.fit(x_train, y_train)

    predicted = text_clf.predict(x_test)

    for j in range(0, len(predicted)):
        pred_df['polarity'][j] = predicted[j] 

    total_test_samples = len(x_test)
    desired_category_index = category_dict[DESIRED_CATEGORY]

    TP = 0 
    for k in range(total_test_samples):
        if pred_df.polarity == 
        if pred_df.predicted_matrix[k][desired_category_index] == 1:
            TP +=1

    acc = TP/total_test_samples
    data_df.at[i, 'baseline'] = str('{0:.2f}'.format(acc*100)) + "%"

predicted_matrix = pred_df.predicted_matrix
category_dict = assign_category(TRAIN_XML_PATH, n)
actual_df = df_something(TEST_XML_PATH, n, category_dict)

a  = []
p = []
for i in range(len(actual_df.matrix)):
    a.append(actual_df.matrix[i].tolist())

for i in range(len(predicted_matrix)):
    p.append(predicted_matrix[i].tolist())



print('---------------')
print('Test Precision: {}'.format(precision_score(a, p, average="macro")))
print('Test Recall: {}'.format(recall_score(a, p, average="macro")))
print('Test Accuracy: {}'.format(accuracy_score(a, p)))
print('---------------')
print('Test F1: {}'.format(f1_score(a, p, average="macro")))


data_df.to_pickle(path.join('polarity', path.join('results', 'data_df.pkl')))

print(data_df)