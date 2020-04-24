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
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np 
import os.path as path 

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




TRAIN_XML_PATH = "ABSA16_Laptops_Train_SB1_v2.xml"
TEST_XML_PATH = "ABSA16_Laptops_Test_GOLD_SB1.xml"


sentences = df_aspect_category(TRAIN_XML_PATH)
categories = Counter(sentences.category).most_common(16)


data_df = pd.DataFrame(columns = ["desired_category", "train_count"])

for i in range(0,16):
    
    DESIRED_CATEGORY = categories[i][0]
    TRAIN_COUNT = categories[i][1]

    print(DESIRED_CATEGORY)
    train_df = df_single_category(TRAIN_XML_PATH, DESIRED_CATEGORY)
    test_df = df_single_category(TRAIN_XML_PATH, DESIRED_CATEGORY)

    train_df_name = 'TRAIN.'+DESIRED_CATEGORY + '.pkl'
    test_df_name = 'TEST.'+DESIRED_CATEGORY + '.pkl'

    train_df.to_pickle(path.join('pandas_data', path.join('aspect_baseline', train_df_name)))
    test_df.to_pickle(path.join('pandas_data', path.join('aspect_baseline', test_df_name)))
    data_df = data_df.append({'desired_category': DESIRED_CATEGORY, 'train_count': TRAIN_COUNT}, ignore_index=True)

data_df.to_pickle(path.join('baseline', path.join('aspect', 'aspect_baseline_data')))

print(data_df)