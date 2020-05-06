from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np 
import os.path as path


def print_stats(y_test, y_pred, model_name):
    """
        Helper function using data from Tensorflow's model evaluation
        function to return the F1 and print model performance stats. 
        Also updates data_df df to contain model acc and f1
    """

    data_df = pd.read_pickle(path.join('subjectivity', path.join('results', 'data_df.pkl')))
    test_f1 = f1_score(y_test, y_pred, average="macro")
    mean = np.mean(y_pred == y_test)

    try:
        best_f1 = data_df[data_df['model']==model_name]['f1'].values[0]
    except: 
        best_f1 = 0 

    if test_f1 > best_f1:
        best_f1 = test_f1   
        data_df = data_df[data_df.model != model_name]
        data_df = data_df.append({'model': model_name, 'acc': mean, 'f1': test_f1}, ignore_index=True)
        print("yes")
        
    data_df.to_pickle(path.join('subjectivity', path.join('results', 'data_df.pkl')))


    print('Test Mean: {}'.format(mean))
    print('---------------')
    print('Test Precision: {}'.format(precision_score(y_test, y_pred, average="macro")))
    print('Test Recall: {}'.format(recall_score(y_test, y_pred, average="macro")))
    print('---------------')
    print('Test F1: {}'.format(test_f1))
    return test_f1

train_df = pd.read_pickle(path.join('subjectivity', path.join('pandas_data', 'TRAIN_SUBJECTIVITY.pkl')))
test_df = pd.read_pickle(path.join('subjectivity', path.join('pandas_data', 'TRAIN_SUBJECTIVITY.pkl')))

x_train, y_train = train_df.text, train_df.subjectivity
x_test, y_test = test_df.text, test_df.subjectivity

text_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
     ('clf', SGDClassifier()),    
     ])

text_clf.fit(x_train, y_train)

predicted = text_clf.predict(x_test)

print_stats(y_test, predicted, 'svm')
# print(confusion_matrix(y_test, predicted))
print(pd.read_pickle(path.join('subjectivity', path.join('results', 'data_df.pkl'))))
