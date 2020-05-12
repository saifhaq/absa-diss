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
from sklearn.svm import SVC

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix, roc_auc_score, multilabel_confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np 
import os.path as path 

def plot_cm(cm,
            target_names,
            title='Confusion matrix',
            cmap=None,
            normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / np.sum(cm).astype('float')
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    thresh = 0.9

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()

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

def print_stats(test_acc, test_precision, test_recall, model_name):
    """
        Helper function using data from Tensorflow's model evaluation
        function to return the F1 and print model performance stats. 
        Also updates data_df df to contain model acc and f1
    """
    test_f1 = 2 * (test_precision * test_recall) / (test_precision + test_recall)

    data_df = pd.read_pickle(path.join('polarity', path.join('results', 'data_df.pkl')))

    try:
        best_acc = data_df[data_df['model']==model_name]['acc'].values[0]
    except: 
        best_acc = 0 

    if test_acc > best_acc:
        data_df = data_df[data_df.model != model_name]
        data_df = data_df.append({'model': model_name, 'acc': test_acc, 'f1': test_f1}, ignore_index=True)
    
    data_df.to_pickle(path.join('polarity', path.join('results', 'data_df.pkl')))

    print('---------------')
    print('Test Accuracy: {}'.format(test_acc))
    print('Test Precision: {}'.format(test_precision))
    print('Test Recall: {}'.format(test_recall))
    print('---------------')
    print('Test F1: {}'.format(test_f1))
    return test_f1



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
        sentence_text =  re.sub(r'[^\w\s]','',sentence_text.lower())

        try: 
            opinions = list(sentence)[1]
            for opinion in opinions:
                if(opinion.attrib['category'] == desired_category):
                    polarity = opinion.attrib['polarity']

                    location = category_dict[opinion.attrib['category']]
                    sentence_id = sentence.attrib['id'] + '__' + str(location)                

                    if(polarity == "positive"):
                        sentences_list.append([sentence_id, sentence_text, 1])

                    elif(polarity == "negative"):
                        sentences_list.append([sentence_id, sentence_text, 0])

        except:
            pass

    return pd.DataFrame(sentences_list, columns = ["id", "text", "polarity"])



def df_predicted(xml_path, n, category_dict):
    """
    """

    tree = et.parse(xml_path)
    reviews = tree.getroot()
    sentences = reviews.findall('**/sentence')

    sentences_list = []

    for sentence in sentences:

        sentence_text = sentence.find('text').text
        sentence_text =  re.sub(r'[^\w\s]','',sentence_text.lower())

        try: 
            opinions = list(sentence)[1]
            for opinion in opinions:
                category_matrix = np.zeros((16, ), dtype=int)

                try:
                    polarity = opinion.attrib['polarity']
                    
                    location = category_dict[opinion.attrib['category']]
                    category_matrix[location] = 1
    
                    sentence_id = sentence.attrib['id'] + '__' + str(location)                

                    if(polarity == "positive"):
                        sentences_list.append([sentence_id, sentence_text, category_matrix, 1, None])

                    elif(polarity == "negative"):
                        sentences_list.append([sentence_id, sentence_text, category_matrix, 0, None])

                except:
                    continue

        except:
            pass


    return pd.DataFrame(sentences_list, columns = ["id", "text", "category", "polarity", "predicted"])


def assign_category(xml_path, n):
    """
        Returns dictionary of the *n* most common as the keys 
        The values of each key is the index of the category  
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

data_df = pd.read_pickle(path.join('polarity', path.join('results', 'data_df.pkl')))

category_dict = assign_category(TRAIN_XML_PATH, n)

pred_df = df_predicted(TEST_XML_PATH, n, category_dict)

stoplist = stoplist()



for i in range(0,n):
    
    DESIRED_CATEGORY = categories[i][0]
    TRAIN_COUNT = categories[i][1]

    train_df = df_single_category(TRAIN_XML_PATH, DESIRED_CATEGORY)
    test_df = df_single_category(TEST_XML_PATH, DESIRED_CATEGORY)


    train_df['text'] = train_df['text'].apply(lambda x: ' '.join([item for item in x.split() if item not in stoplist]))
    test_df['text'] = test_df['text'].apply(lambda x: ' '.join([item for item in x.split() if item not in stoplist]))

    train_df_name = 'BASELINE_'+'TRAIN_'+DESIRED_CATEGORY + '.pkl'
    test_df_name =  'BASELINE_'+'TEST_'+DESIRED_CATEGORY + '.pkl'

    train_df.to_pickle(path.join('polarity', path.join('pandas_data', train_df_name)))
    test_df.to_pickle(path.join('polarity', path.join('pandas_data', test_df_name)))


    x_train, y_train = train_df.text, train_df.polarity

    text_clf = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('svc', SGDClassifier()),    
        ])

    text_clf.fit(x_train, y_train)

    predicted = text_clf.predict(test_df.text)

    for j in range(0, len(predicted)):
        pred_df.loc[(pred_df['id'] == test_df['id'].iat[j]), ['predicted']] = predicted[j]



a = pred_df.polarity.to_list()
p = pred_df.predicted.to_list()

test_acc = accuracy_score(a, p)
test_precision = precision_score(a, p)
test_recall = recall_score(a, p)

class_names = ['Negative', 'Positive']

model_name = 'svm_advanced'

test_f1 = print_stats(test_acc, test_precision, test_recall, model_name)
cm = confusion_matrix(a, p)

title = "Advanced SVM Polarity Classifier: Normalized Confusion Matrix"
plot_cm(cm, class_names, title=title)


# class_names = ['Negative', 'Positive']

# model_name = 'svm'

# pred_labels = (predicted > 0.5).astype(np.int)

# cm = confusion_matrix(y_test, pred_labels)

# title = "Basic SVM Polarity Classifier: Normalized Confusion Matrix"
# plot_cm(cm, class_names, title=title)



print(pd.read_pickle(path.join('polarity', path.join('results', 'data_df.pkl'))))

