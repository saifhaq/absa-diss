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
import xml.etree.ElementTree as et

def df_aspect_category(xml_path):
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
        Returns dictionary of n most common categories 
    """

    sentences = df_aspect_category(xml_path)
    categories = Counter(sentences.category).most_common(n)

    common_categories = [category_tuple[0] for category_tuple in categories]

    common_df = sentences[sentences['category'].isin(common_categories)]

    assigned = {}

    for i in range(0, len(common_categories)):
        assigned[common_categories[i]] = i 

    return assigned
    # common_df[''] = common_df[]

    return None

def df_categories(xml_path, n, category_dict, empty_matrix_wanted = True):
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
                location = category_dict[opinion.attrib['category']]
                try:
                    category_matrix[location] = 1
                except: 
                    pass
                # category_matrix[i] = assigned(category_dict[opinion.attrib['category']])
            z = np.count_nonzero(category_matrix)
            if (not empty_matrix_wanted):
                if (z!=0):
                    sentences_list.append([sentence_id, sentence_text, categories, category_matrix])
            else:
                # if category_matrix[0] == 1:
                sentences_list.append([sentence_id, sentence_text, categories, category_matrix])

        except:
            pass

    return pd.DataFrame(sentences_list, columns = ["id", "text", "category", "matrix"])


category_dict = {'LAPTOP#GENERAL': 0, 'LAPTOP#OPERATION_PERFORMANCE': 1, 'LAPTOP#DESIGN_FEATURES': 2, 'LAPTOP#QUALITY': 3, 'LAPTOP#MISCELLANEOUS': 4, 'LAPTOP#USABILITY': 5, 'SUPPORT#QUALITY': 6, 'LAPTOP#PRICE': 7, 'COMPANY#GENERAL': 8, 'BATTERY#OPERATION_PERFORMANCE': 9, 'LAPTOP#CONNECTIVITY': 10, 'DISPLAY#QUALITY': 11, 'LAPTOP#PORTABILITY': 12, 'OS#GENERAL': 13, 'SOFTWARE#GENERAL': 14, 'KEYBOARD#DESIGN_FEATURES': 15}

TRAIN_XML_PATH = "ABSA16_Laptops_Train_SB1_v2.xml"
TEST_XML_PATH = "ABSA16_Laptops_Test_GOLD_SB1.xml"

category_dict = assign_category(TRAIN_XML_PATH, 100)


train_df = df_categories(TRAIN_XML_PATH, 100, category_dict, True)
test_df = df_categories(TEST_XML_PATH, 100, category_dict, True)

print(train_df)
# category_dict = df_categories 
# print(train_df)
# category_dict = assign_category(TRAIN_XML_PATH, 8)

# sentences = df_aspect_category(TEST_XML_PATH)
# categories = Counter(sentences.category).most_common(50)
# print(categories)

# print(category_dict)

# train_df = pd.read_pickle(path.join('pandas_data', 'aspect_category_detection_train.pkl'))
# test_df = pd.read_pickle(path.join('pandas_data', 'aspect_category_detection_test.pkl'))

# print(test_df)

# for index, row in test_df.iterrows():
#     # non_zeroes = np.count_nonzero(row['matrix'])
#     print(index)

