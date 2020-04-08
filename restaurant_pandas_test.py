import pandas as pd
import xml.etree.ElementTree as et
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn import svm
import numpy as np 
import os.path as path
import re
from nltk.corpus import stopwords
import nltk


def sentence_categories(xml_path):
    """
        Returns lookup dictionary that has sentence id's categories
    """
    tree = et.parse(xml_path)
    reviews = tree.getroot()
    sentences = reviews.findall('**/sentence')

    sentence_categories = {}

    for sentence in sentences:
        for opinions in sentence:
            categories = []
            for opinion in opinions:
                categories.append(opinion.attrib['category'])

        sentence_categories[sentence.attrib['id']] = categories
    return sentence_categories


def pre_process_sentence(text):
    """
    Strips, removes
    """
    processed = re.sub(r'[^\w\s]','',df.text.lower())
    return(processed)

def df_sentences(xml_path):
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

        try: 
            opinions = list(sentence)[1]

            for opinion in opinions:
                category = opinion.attrib['category']
                polarity = opinion.attrib['polarity']
                sentences_list.append([sentence_id, sentence_text, category, polarity])

        # except:
        #     polarity = 'None'
        #     category = 'None'
        #     sentences_list.append([sentence_id, sentence_text, category, polarity])

        except:
            pass

    return pd.DataFrame(sentences_list, columns = ["id", "text", "category", "polarity"])


def tokenizer(df):
    """
        Takes a sentence, and reuturns a matrix
    """

    df.text



def df_subjectivity(xml_path):
    """
        Takes XML Training data and returns a pandas dataframe of unique sentences;
        with subjectivity 1 if they express an opinion, 0 if not 
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

            num_opinions = 0 
            count_subjectivity = 0
            for opinion in opinions:
                polarity = opinion.attrib['polarity']
                num_opinions+=1
                if polarity == "positive" or polarity == "negative":
                    count_subjectivity += 1
            
            if (count_subjectivity !=0):
                sentences_list.append([sentence_id, sentence_text, 1])
            else:
                sentences_list.append([sentence_id, sentence_text, 0])

        except:
            # Ignore sentences that do have any opinions labelled
            pass



    return pd.DataFrame(sentences_list, columns = ["id", "text", "subjectivity"])


def df_test_subjectivity(xml_path):
    """
        Takes XML Test data and returns a pandas dataframe of sentences;
    """

    tree = et.parse(xml_path)
    reviews = tree.getroot()
    sentences = reviews.findall('**/sentence')

    sentences_list = []

    for sentence in sentences:

        sentence_id = sentence.attrib['id']                
        sentence_text = sentence.find('text').text
        sentences_list.append([sentence_id, sentence_text])



    return pd.DataFrame(sentences_list, columns = ["id", "text"])



def df_polarity(xml_path):
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

        try: 
            opinions = list(sentence)[1]

            for opinion in opinions:
                category = opinion.attrib['category']
                polarity = opinion.attrib['polarity']
                
                if(polarity == "positive"):
                    polarity_val = 1
                elif(polarity == "negative"):
                    polarity_val = 0 
                sentences_list.append([sentence_id, sentence_text, polarity_val])

        except Exception as se:
            # print('Uhoh, got SomeException:' + se.args[0])
            pass


    return pd.DataFrame(sentences_list, columns = ["id", "text", "polarity"])




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


    
def df_categories_baseline(xml_path, n, category_dict, empty_matrix_wanted = True):
    """
        Takes XML Training data and returns a pandas dataframe of sentences;
    """

    tree = et.parse(xml_path)
    reviews = tree.getroot()
    sentences = reviews.findall('**/sentence')

    sentences_list = []
    
    # category_dict = assign_category(xml_path, n)

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
                    sentences_list.append([sentence_id, sentence_text, categories, category_matrix[0]])
            else:
                sentences_list.append([sentence_id, sentence_text, categories, category_matrix[0]])

        except:
            pass

    return pd.DataFrame(sentences_list, columns = ["id", "text", "category", "matrix"])



def df_categories(xml_path, n, category_dict, empty_matrix_wanted = True):
    """
        Takes XML Training data and returns a pandas dataframe of sentences;
    """

    tree = et.parse(xml_path)
    reviews = tree.getroot()
    sentences = reviews.findall('**/sentence')

    sentences_list = []
    
    # category_dict = assign_category(xml_path, n)

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


TRAIN_XML_PATH = "ABSA16_Restaurants_Train_SB1_v2.xml"
TEST_XML_PATH = "EN_REST_SB1_TEST.xml.gold"

# category_dict = {'LAPTOP#GENERAL': 0, 'LAPTOP#OPERATION_PERFORMANCE': 1, 'LAPTOP#DESIGN_FEATURES': 2, 'LAPTOP#QUALITY': 3, 'LAPTOP#MISCELLANEOUS': 4, 'LAPTOP#USABILITY': 5, 'SUPPORT#QUALITY': 6, 'LAPTOP#PRICE': 7, 'COMPANY#GENERAL': 8, 'BATTERY#OPERATION_PERFORMANCE': 9, 'LAPTOP#CONNECTIVITY': 10, 'DISPLAY#QUALITY': 11, 'LAPTOP#PORTABILITY': 12, 'OS#GENERAL': 13, 'SOFTWARE#GENERAL': 14, 'KEYBOARD#DESIGN_FEATURES': 15}
category_dict = {'FOOD#QUALITY': 0, 'SERVICE#GENERAL': 1, 'RESTAURANT#GENERAL': 2, 'AMBIENCE#GENERAL': 3, 'FOOD#STYLE_OPTIONS': 4, 'RESTAURANT#MISCELLANEOUS': 5, 'FOOD#PRICES': 6, 'RESTAURANT#PRICES': 7}

train_df = df_categories(TRAIN_XML_PATH, 8, category_dict, True)

# cats = assign_category(TRAIN_XML_PATH, 8)
# print(cats)
print(train_df)
test_df = df_categories(TEST_XML_PATH, 8, category_dict, True)

stop = stopwords.words('english')

# train_df['text'] = train_df['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
# test_df['text'] = test_df['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

train_df.to_pickle(path.join('pandas_data', 'restaurants_aspect_category_detection_train.pkl'))
test_df.to_pickle(path.join('pandas_data', 'restaurants_aspect_category_detection_test.pkl'))

