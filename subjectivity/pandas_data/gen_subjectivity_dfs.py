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

def df_subjectivity(xml_path):
    """
        Takes XML Training data and returns a pandas dataframe of unique sentences;
        with subjectivity as [1,0] if they express an opinion, [0,1] if not 
    """

    tree = et.parse(xml_path)
    reviews = tree.getroot()
    sentences = reviews.findall('**/sentence')

    sentences_list = []

    for sentence in sentences:

        sentence_id = sentence.attrib['id']                
        sentence_text = sentence.find('text').text
        sentence_text =  re.sub(r'[^\w\s]','',sentence_text.lower())
        count_subjectivity = 0

        try: 
            try:
                opinions = list(sentence)[1]
                for opinion in opinions:
                    polarity = opinion.attrib['polarity']
                    if polarity == "positive" or polarity == "negative":
                        count_subjectivity += 10
    
            except:
                pass
            if (count_subjectivity ==0):
                sentences_list.append([sentence_id, sentence_text, [1,0]])
            elif count_subjectivity >0:
                sentences_list.append([sentence_id, sentence_text, [0,1]])

        except:
            pass



    return pd.DataFrame(sentences_list, columns = ["id", "text", "subjectivity"])


TRAIN_XML_PATH = "ABSA16_Laptops_Train_SB1_v2.xml"
TEST_XML_PATH = "ABSA16_Laptops_Test_GOLD_SB1.xml"


train_df = df_subjectivity(TRAIN_XML_PATH)
test_df = df_subjectivity(TEST_XML_PATH)

train_df_name = 'TRAIN_SUBJECTIVITY.pkl'
test_df_name =  'TEST_SUBJECTIVITY.pkl'

print(train_df)

# train_df.to_pickle(path.join('subjectivity', path.join('pandas_data', train_df_name)))
# test_df.to_pickle(path.join('subjectivity', path.join('pandas_data', test_df_name)))

