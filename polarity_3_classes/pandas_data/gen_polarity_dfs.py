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

def df_sentences(xml_path):
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

def assign_category(xml_path, n):
    """
        Returns dictionary of the *n* most common as the keys 
        The values of each key is the index of the category  
    """

    sentences = df_sentences(xml_path)
    categories = Counter(sentences.category).most_common(n)
    common_categories = [category_tuple[0] for category_tuple in categories]
    assigned = {}

    for i in range(0, len(common_categories)):
        assigned[common_categories[i]] = i 

    return assigned

def df_polarity(xml_path, n, category_dict):
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
                        sentences_list.append([sentence_id, sentence_text, category_matrix, [1, 0, 0]])
                    elif(polarity == "negative"):
                        sentences_list.append([sentence_id, sentence_text, category_matrix, [0, 1, 0]])
                    elif(polarity == "neutral"):
                        sentences_list.append([sentence_id, sentence_text, category_matrix, [0, 0, 1]])
                                
                except:
                    continue

        except:
            pass


    return pd.DataFrame(sentences_list, columns = ["id", "text", "category", "polarity"])


TRAIN_XML_PATH = "ABSA16_Laptops_Train_SB1_v2.xml"
TEST_XML_PATH = "ABSA16_Laptops_Test_GOLD_SB1.xml"

category_dict = assign_category(TRAIN_XML_PATH, 16)

train_df = df_polarity(TRAIN_XML_PATH, 16, category_dict)
test_df = df_polarity(TEST_XML_PATH, 16, category_dict)

train_df_name = 'TRAIN_POLARITY.pkl'
test_df_name =  'TEST_POLARITY.pkl'

print(test_df)

train_df.to_pickle(path.join('polarity', path.join('pandas_data', train_df_name)))
test_df.to_pickle(path.join('polarity', path.join('pandas_data', test_df_name)))

