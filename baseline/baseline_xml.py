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


def stoplist(file_name = "stopwords.txt"):
  stopwords_txt = open(path.join('preprocessing', file_name))
  stoplist = []
  for line in stopwords_txt:
      values = line.split()
      stoplist.append(values[0])
  stopwords_txt.close()
  return stoplist

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

def df_only_single_category(xml_path, desired_category):
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

def df_predicted_category(xml_path, n):
    """

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
            matrix = np.zeros((n,), dtype=int)
            sentences_list.append([sentence_id, sentence_text, matrix])

        except:
            pass

    return pd.DataFrame(sentences_list, columns = ["id", "text", "predicted_matrix"])


def df_something(xml_path, n, category_dict, empty_matrix_wanted = True):
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

def assign_category(xml_path, n):
    """
        Returns dictionary of n most common categories 
    """

    sentences = df_aspect_category(xml_path)
    categories = Counter(sentences.category).most_common(n)
    common_categories = [category_tuple[0] for category_tuple in categories]
    assigned = {}

    for i in range(0, len(common_categories)):
        assigned[common_categories[i]] = i 

    return assigned

TRAIN_XML_PATH = "ABSA16_Laptops_Train_SB1_v2.xml"
TEST_XML_PATH = "ABSA16_Laptops_Test_GOLD_SB1.xml"

n = 16

sentences = df_aspect_category(TRAIN_XML_PATH)
categories = Counter(sentences.category).most_common(n)

sentences_test = df_aspect_category(TRAIN_XML_PATH)
categories_test = Counter(sentences.category).most_common(n)

data_df = pd.DataFrame(columns = ["desired_category", "train_count"])

pred_df = df_predicted_category(TEST_XML_PATH, n)

# print(categories)
DESIRED_CATEGORY = categories[0][0]

stoplist = stoplist()
for i in range(0,n):
    
    DESIRED_CATEGORY = categories[i][0]
    TRAIN_COUNT = categories[i][1]

    train_df = df_single_category(TRAIN_XML_PATH, DESIRED_CATEGORY)
    test_df = df_single_category(TEST_XML_PATH, DESIRED_CATEGORY)
    test_single_df = df_only_single_category(TEST_XML_PATH, DESIRED_CATEGORY)

    

    train_df['text'] = train_df['text'].apply(lambda x: ' '.join([item for item in x.split() if item not in stoplist]))
    test_df['text'] = test_df['text'].apply(lambda x: ' '.join([item for item in x.split() if item not in stoplist]))

    train_df_name = 'TRAIN.'+DESIRED_CATEGORY + '.pkl'
    test_df_name = 'TEST.'+DESIRED_CATEGORY + '.pkl'

    train_df.to_pickle(path.join('pandas_data', path.join('aspect_baseline', train_df_name)))
    test_df.to_pickle(path.join('pandas_data', path.join('aspect_baseline', test_df_name)))


    x_train, y_train = train_df.text, train_df.desired_category
    x_test, y_test = test_df.text, test_df.desired_category

    text_clf = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('svc', SGDClassifier()),    
        ])

    text_clf.fit(x_train, y_train)

    predicted = text_clf.predict(x_test)

    for j in range(0, len(predicted)):
        matrix = pred_df['predicted_matrix'][j] 
        matrix[i] = predicted[j] 

    total_test_samples = len(x_test)
    category_dict = assign_category(TRAIN_XML_PATH, 16)
    desired_category_index = category_dict[DESIRED_CATEGORY]

    TP = 0 
    for i in range(total_test_samples):
        if pred_df.predicted_matrix[i][desired_category_index] == 1:
            TP +=1



    mean = np.mean(predicted == y_test)
    acc = accuracy_score(y_test, predicted)
    CM = confusion_matrix(y_test, predicted)

    print("----------")
    print(CM)
    print("----------")

    predicted_individual = text_clf.predict(test_single_df.text)
    actual_individual = test_single_df.desired_category
    acc2 = f1_score(actual_individual, predicted_individual)

    acc = TP/total_test_samples

    data_df = data_df.append({'desired_category': DESIRED_CATEGORY, 'train_count': TRAIN_COUNT, 'acc': acc}, ignore_index=True)

predicted_matrix = pred_df.predicted_matrix
# actual_df = pd.read_pickle(path.join('pandas_data', 'aspect_category_detection_test_'+str(n)+'_classes.pkl'))
# actual_df = df_actual(TEST_XML_PATH, n, categories_test)
# category_dict = {'LAPTOP#GENERAL': 0, 'LAPTOP#OPERATION_PERFORMANCE': 1, 'LAPTOP#DESIGN_FEATURES': 2, 'LAPTOP#QUALITY': 3, 'LAPTOP#MISCELLANEOUS': 4, 'LAPTOP#USABILITY': 5, 'SUPPORT#QUALITY': 6, 'LAPTOP#PRICE': 7, 'COMPANY#GENERAL': 8, 'BATTERY#OPERATION_PERFORMANCE': 9, 'LAPTOP#CONNECTIVITY': 10, 'DISPLAY#QUALITY': 11, 'LAPTOP#PORTABILITY': 12, 'OS#GENERAL': 13, 'SOFTWARE#GENERAL': 14, 'KEYBOARD#DESIGN_FEATURES': 15}
category_dict = assign_category(TRAIN_XML_PATH, n)
actual_df = df_something(TEST_XML_PATH, n, category_dict, True)

print(actual_df)
actual_matrix = actual_df.matrix


pred= np.reshape(predicted_matrix.values, (predicted_matrix.shape[0]))
# s = [a[0] for a in pred]

# print(actual_matrix[0])
#array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])]
# print(len(actual_matrix))
a  = []
p = []
for i in range(len(actual_matrix)):
    a.append(actual_matrix[i].tolist())

for i in range(len(predicted_matrix)):
    p.append(predicted_matrix[i].tolist())

# print(a)
# print(np.asarray(p).argmax(axis=1))
# print(multilabel_confusion_matrix(a, p))

print('---------------')
print('Test Precision: {}'.format(precision_score(a, p, average="macro")))
print('Test Recall: {}'.format(recall_score(a, p, average="macro")))
print('Test Accuracy: {}'.format(accuracy_score(a, p)))
print('---------------')
print('Test F1: {}'.format(f1_score(a, p, average="macro")))



# ---------------
# Test Precision: 0.47925956844421036
# Test Recall: 0.46860217414676975
# Test Accuracy: 0.5160891089108911
# ---------------
# Test F1: 0.46868394423516574

# f1 = f1_score(a, p, average="macro")
# print(f1)
# print(a)
# print(actual_matrix.head(5).array)
# print(predicted_matrix.head(5).array)
# df.as_matrix(columns=[df[1:]])

# print(pred_df.text.head(5))
# print(actual_df.text.head(5))
# print(pred_df.at[430, 'text'])
# print(pred_df.at[430, 'predicted_matrix'])

# print(pred_df.head(1))

# data_df.to_pickle(path.join('baseline', path.join('aspect', 'aspect_baseline_data')))

print(data_df)