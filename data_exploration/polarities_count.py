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


def df_categories(xml_path):
    """
        Takes *xml_path* and returns dataframe of each sentence and their categories. 
        Each sentence is returned once.  
        
        Dataframe returned as: [id, text, category, polarity] 
    """

    tree = et.parse(xml_path)
    reviews = tree.getroot()
    sentences = reviews.findall('**/sentence')

    sentences_list = []
    
    for sentence in sentences:

        sentence_id = sentence.attrib['id']                
        sentence_text = sentence.find('text').text
        sentence_text =  re.sub(r'[^\w\s]','',sentence_text.lower())
        categories = []
        polarities = []

        try: 
            opinions = list(sentence)[1]
            for opinion in opinions:
                categories.append(opinion.attrib['category'])
                polarities.append(opinion.attrib['polarity'])
        
        except:
            pass
        
        sentences_list.append([sentence_id, sentence_text, categories, polarities])


    return pd.DataFrame(sentences_list, columns = ["id", "text", "category", "polarity"])

TRAIN_XML_PATH = "ABSA16_Laptops_Train_SB1_v2.xml"
TEST_XML_PATH = "ABSA16_Laptops_Test_GOLD_SB1.xml"


train_df = df_categories(TRAIN_XML_PATH)
test_df = df_categories(TEST_XML_PATH)


polarities_df = pd.DataFrame(columns = ["Polarity", "Count", "Percentage"])

polarities_df = polarities_df.append({'Polarity': 'Neutral', 'Count': 0}, ignore_index=True)
polarities_df = polarities_df.append({'Polarity': 'Positive', 'Count': 0}, ignore_index=True)
polarities_df = polarities_df.append({'Polarity': 'Negative', 'Count': 0}, ignore_index=True)


print(train_df)
 
# for i in range(len(train_df.category)):
#     index = len(train_df.category[i])
#     if index >=4:
#         index = 4
#     try:
#         opinions_df.loc[index,'Count'] +=1
#     except:
#         opinions_df.loc[index,'Count'] = 1

# n_samples = len(train_df)

# for index, row in opinions_df.iterrows():
#     p = (row['Count'] / n_samples) *100
    
#     row['Percentage'] = str('{0:.2f}'.format(p)) + "%"

print(polarities_df)

# opinions_df.to_pickle(path.join('data_exploration', path.join('results', 'opinions_df.pkl')))

# polarities_df = pd.read_pickle(path.join('data_exploration', path.join('results', 'polarities_df.pkl')))
