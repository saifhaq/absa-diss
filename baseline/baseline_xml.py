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
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix, roc_auc_score
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


def df_predicted_category(xml_path, n):
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
            matrix = np.zeros((n,), dtype=int)
            sentences_list.append([sentence_id, sentence_text, matrix])

        except:
            pass

    return pd.DataFrame(sentences_list, columns = ["id", "text", "predicted_matrix"])


TRAIN_XML_PATH = "ABSA16_Laptops_Train_SB1_v2.xml"
TEST_XML_PATH = "ABSA16_Laptops_Test_GOLD_SB1.xml"

n = 8
sentences = df_aspect_category(TRAIN_XML_PATH)
categories = Counter(sentences.category).most_common(n)

sentences_test = df_aspect_category(TRAIN_XML_PATH)
categories_test = Counter(sentences.category).most_common(n)

data_df = pd.DataFrame(columns = ["desired_category", "train_count"])
# predicted_category_matrix =
# df = sentences.append({'predicted': predicted_category_matrix}, ignore_index=True)
# print(df)


pred_df = df_predicted_category(TEST_XML_PATH, n)


DESIRED_CATEGORY = categories[0][0]


# matrix = df['predicted_matrix'][2] 
# matrix[0] = 1 

# print(len(df))


# sentences.assign(Name='predicted')
# s = np.zeros((16), dtype=int)
# sentences['predicted'] =  s
stoplist = stoplist()
for i in range(0,n):
    
    DESIRED_CATEGORY = categories[i][0]
    TRAIN_COUNT = categories[i][1]

    train_df = df_single_category(TRAIN_XML_PATH, DESIRED_CATEGORY)
    test_df = df_single_category(TEST_XML_PATH, DESIRED_CATEGORY)

    

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

    mean = np.mean(predicted == y_test)
    acc = roc_auc_score(y_test, predicted)

    data_df = data_df.append({'desired_category': DESIRED_CATEGORY, 'train_count': TRAIN_COUNT, 'acc': acc}, ignore_index=True)

predicted_matrix = pred_df.predicted_matrix
actual_df = pd.read_pickle(path.join('pandas_data', 'aspect_category_detection_test_8_classes.pkl'))
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

# print(data_df)