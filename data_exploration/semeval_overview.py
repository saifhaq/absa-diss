import pandas as pd
import os.path as path
import xml.etree.ElementTree as et
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np 
import re

def df_semeval(xml_path):
    """
        Takes *xml_path* and returns dataframe of each sentence with array of labels, being an 
        array of tuples: [category, polarity]. 
        Sentence returned once for multiple categories 
        
        Dataframe returned as: [id, text, [[category, polarity], ...]] 
    """

    tree = et.parse(xml_path)
    reviews = tree.getroot()
    sentences = reviews.findall('**/sentence')

    sentences_list = []

    for sentence in sentences:
        labels = []

        sentence_id = sentence.attrib['id']                
        sentence_text = sentence.find('text').text
        sentence_text =  re.sub(r'[^\w\s]','',sentence_text.lower())

        try: 
            opinions = list(sentence)[1]

            for opinion in opinions:
                category = opinion.attrib['category']
                polarity = opinion.attrib['polarity']
                labels.append([category, polarity])
            
            sentences_list.append([sentence_id, sentence_text, labels])

        except:
            sentences_list.append([sentence_id, sentence_text, []])
            pass

    return pd.DataFrame(sentences_list, columns = ["id", "text", "labels"])



TRAIN_XML_PATH = "ABSA16_Laptops_Train_SB1_v2.xml"
TEST_XML_PATH = "ABSA16_Laptops_Test_GOLD_SB1.xml"

train_df = df_semeval(TRAIN_XML_PATH)
test_df = df_semeval(TEST_XML_PATH)

n_train_sentences = len(train_df) 
n_test_sentences = len(test_df) 

n_train_tuples = len([item for sublist in train_df.labels for item in sublist])
n_test_tuples = len([item for sublist in test_df.labels for item in sublist])


print(n_train_sentences)
print("--")
print(n_test_sentences)
print("--")
print(n_train_tuples)
print("--")
print(n_test_tuples)

train_df.text.to_list()

# total_avg = sum( map(len, train_df.text) ) / len(train_df.text)

# print(total_avg)
# print(train_df.text.to_list())