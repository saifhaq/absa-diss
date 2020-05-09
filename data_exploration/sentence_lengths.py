import pandas as pd
import os.path as path
import xml.etree.ElementTree as et
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np 
import re

def stoplist(file_name = "stopwords.txt"):
    """
        Returns an array of stopwords, from each line of the 
        *file_name* text file
    """
    stopwords_txt = open(path.join('preprocessing', file_name))
    stoplist = []
    for line in stopwords_txt:
        values = line.split()
        stoplist.append(values[0])
    stopwords_txt.close()
    return stoplist

def df_semeval(xml_path, stoplist):
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

train_df = df_semeval(TRAIN_XML_PATH, False)
test_df = df_semeval(TEST_XML_PATH, False)




stopwords = stoplist()

train_text_array = train_df['text'].apply(lambda x: [item for item in x.split()])
test_text_array = test_df['text'].apply(lambda x: [item for item in x.split()])

train_text_array_stoplist = train_df['text'] = test_df['text'].apply(lambda x: [item for item in x.split() if item not in stopwords])
test_text_array_stoplist = test_df['text'] = test_df['text'].apply(lambda x: [item for item in x.split() if item not in stopwords])

train_avg_words = sum( map(len, train_text_array) ) / len(train_text_array)
# test_avg_words = sum( map(len, test_text_array) ) / len(test_text_array)

train_avg_words_stoplist = sum( map(len, train_text_array_stoplist) ) / len(train_text_array_stoplist)
# test_avg_words_stoplist = sum( map(len, test_avg_words) ) / len(test_avg_words)

print(train_avg_words_stoplist)
# print(train_avg_words_stoplist)




# train_avg_words_after_stoplist = sum( map(len, train_df.text) ) / len(train_df.text)
# test_avg_words_after_stoplist = sum( map(len, test_df.text) ) / len(test_df.text)

# print(train_avg_words_after_stoplist)
