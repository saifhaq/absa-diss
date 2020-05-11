import pandas as pd
import os.path as path
import xml.etree.ElementTree as et
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np 
import re
import itertools

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


stopping = True

if stopping:
    stopwords = stoplist()
    train_sentences = train_df['text'] = train_df['text'].apply(lambda x: [item for item in x.split() if item not in stopwords])
    test_sentences = test_df['text'] = test_df['text'].apply(lambda x: [item for item in x.split() if item not in stopwords])
    
else:
    train_sentences = train_df['text'].apply(lambda x: [item for item in x.split()])
train_words = []

sentence_lengths = []
for index, row in train_df.iterrows():
    sentence_lengths.append(len(row.text))

print(len(sentence_lengths))
print(np.mean(sentence_lengths))

print(train_sentences[2498])
# 5.0924 with stopping
# 5.0924 with stopping