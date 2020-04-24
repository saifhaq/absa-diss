import pandas as pd
import xml.etree.ElementTree as et
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn import svm
import numpy as np 
import os.path as path
import re




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


    
def df_single_category(xml_path, desired_category):
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
        label = 0
        sentence_text =  re.sub(r'[^\w\s]','',sentence_text.lower())

        try: 
            opinions = list(sentence)[1]
            categories = []
            for opinion in opinions:
                if(opinion.attrib['category'] == desired_category):
                    label = 1
                # category_matrix[i] = assigned(category_dict[opinion.attrib['category']])
            
            sentences_list.append([sentence_id, sentence_text, label])

        except:
            pass

    return pd.DataFrame(sentences_list, columns = ["id", "text", "desired_category"])



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


def df_something(xml_path, n, category_dict, empty_matrix_wanted = True):
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
                try:
                    location = category_dict[opinion.attrib['category']]
                    category_matrix[location] = 1
                except:
                    continue

                # category_matrix[i] = assigned(category_dict[opinion.attrib['category']])
            sentences_list.append([sentence_id, sentence_text, categories, category_matrix])

        except:
            pass

    return pd.DataFrame(sentences_list, columns = ["id", "text", "category", "matrix"])

TRAIN_XML_PATH = "ABSA16_Laptops_Train_SB1_v2.xml"
TEST_XML_PATH = "ABSA16_Laptops_Test_GOLD_SB1.xml"


# train_df = df_polarity(TRAIN_XML_PATH)
# test_df = df_polarity(TEST_XML_PATH)
# train_df.to_pickle(path.join('pandas_data', 'polarity_train.pkl'))
# test_df.to_pickle(path.join('pandas_data', 'polarity_test.pkl'))

# train_df = df_subjectivity(TRAIN_XML_PATH)
# test_df = df_subjectivity(TEST_XML_PATH)
# train_df.to_pickle(path.join('pandas_data', 'subjectivity_train.pkl'))
# test_df.to_pickle(path.join('pandas_data', 'subjectivity_test.pkl'))

category_dict = {'LAPTOP#GENERAL': 0, 'LAPTOP#OPERATION_PERFORMANCE': 1, 'LAPTOP#DESIGN_FEATURES': 2, 'LAPTOP#QUALITY': 3, 'LAPTOP#MISCELLANEOUS': 4, 'LAPTOP#USABILITY': 5, 'SUPPORT#QUALITY': 6, 'LAPTOP#PRICE': 7, 'COMPANY#GENERAL': 8, 'BATTERY#OPERATION_PERFORMANCE': 9, 'LAPTOP#CONNECTIVITY': 10, 'DISPLAY#QUALITY': 11, 'LAPTOP#PORTABILITY': 12, 'OS#GENERAL': 13, 'SOFTWARE#GENERAL': 14, 'KEYBOARD#DESIGN_FEATURES': 15}
train_df = df_something(TRAIN_XML_PATH, 8, category_dict, True)
test_df = df_something(TEST_XML_PATH, 8, category_dict, True)
train_df.to_pickle(path.join('pandas_data', 'aspect_category_detection_train_8_classes.pkl'))
test_df.to_pickle(path.join('pandas_data', 'aspect_category_detection_test_8_classes.pkl'))

print(len(test_df))
# print(test_df)
# print(test_df.at[430, 'text'])
# print(test_df.at[430, 'matrix'])

# train_df = df_polarity(TRAIN_XML_PATH)


# train_df = df_single_category(TRAIN_XML_PATH, 'LAPTOP#OPERATION_PERFORMANCE')
# df2 = train_df.loc[train_df['desired_category'] == 1]


# print(train_df)
# TRAIN_XML_PATH = (path.join('xml_data', "example_review.xml"))
# # train_df = df_subjectivity(TRAIN_XML_PATH, 10, category_dict, False)
# train_df = df_subjectivity(TRAIN_XML_PATH)

# print(train_df.head(10))

# train_df.to_pickle(path.join('pandas_data', 'example_review_subjectivity.pkl'))



# train_df = df_categories(TRAIN_XML_PATH, 8, category_dict, True)
# test_df = df_categories(TEST_XML_PATH, 8, category_dict, True)


# train_df.to_pickle(path.join('pandas_data', 'aspect_category_detection_train.pkl'))
# test_df.to_pickle(path.join('pandas_data', 'aspect_category_detection_test.pkl'))

# print(len(train_df))
# print(len(test_df))
# assigned = assign_category(TRAIN_XML_PATH, 10)
# sentences = df_aspect_category(TEST_XML_PATH)
# categories = Counter(sentences.category).most_common(10)

# print(categories)

# print(test_df)

# [('LAPTOP#GENERAL', 634), ('LAPTOP#OPERATION_PERFORMANCE', 278), ('LAPTOP#DESIGN_FEATURES', 253), ('LAPTOP#QUALITY', 224), ('LAPTOP#MISCELLANEOUS', 142), ('LAPTOP#USABILITY', 141), ('SUPPORT#QUALITY', 138), ('LAPTOP#PRICE', 136), ('COMPANY#1', 90), ('BATTERY#OPERATION_PERFORMANCE', 86)]

# train_df = df_categories(TRAIN_XML_PATH, 6, category_dict, True)
# test_df = df_categories(TEST_XML_PATH, 6, category_dict, True)

# train_df.to_pickle(path.join('pandas_data', 'aspect_category_detection_train.pkl'))
# test_df.to_pickle(path.join('pandas_data', 'aspect_category_detection_test.pkl'))

# print(len(train_df))
# print(train_df)

# pos =  train_df.loc[(train_df["matrix"] == 1)]
# print(pos.tail(30))

# df = df_categories(XML_PATH, n=8)
# print(len(df))
# df.to_pickle('aspect_category_detection_train_10_classes.pkl')


# df = df_categories(XML_SB1_TEST_GOLD_PATH, n=8)
# print(df)
# df.to_pickle('aspect_category_detection_test_10_classes.pkl')


# from sklearn.metrics import confusion_matrix
# from sklearn.pipeline import Pipeline
# from sklearn.linear_model import SGDClassifier
# from sklearn.feature_extraction.text import TfidfTransformer
# from sklearn.metrics import confusion_matrix
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix

# train_df = df_categories(XML_PATH)
# test_df = df_categories(XML_SB1_TEST_GOLD_PATH)

# x_train, y_train = train_df.text, train_df.matrix
# x_test, y_test = test_df.text, test_df.matrix

# text_clf = Pipeline([
#     ('vect', CountVectorizer()),
#     ('tfidf', TfidfTransformer()),
#      ('clf', SGDClassifier()),    
#      ])

# text_clf.fit(x_train, y_train)

# predicted = text_clf.predict(x_test)
# mean = np.mean(predicted == y_test)
# print(mean)

# print(f1_score(y_test, predicted, average="macro"))
# print(precision_score(y_test, predicted, average="macro"))
# print(recall_score(y_test, predicted, average="macro"))   
# # print(confusion_matrix(y_test, predicted))