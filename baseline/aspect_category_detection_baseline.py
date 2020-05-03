from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
import sklearn
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np 
import os.path as path 

DESIRED_CATEGORY = 'LAPTOP#GENERAL'

train_df_name = 'TRAIN.'+DESIRED_CATEGORY + '.pkl'
test_df_name = 'TEST.'+DESIRED_CATEGORY + '.pkl'

train_df = pd.read_pickle(path.join('pandas_data', path.join('aspect_baseline', train_df_name)))
test_df = pd.read_pickle(path.join('pandas_data', path.join('aspect_baseline', test_df_name)))

# train_df = pd.read_pickle(path.join('pandas_data', 'aspect_category_detection_train.pkl'))
# test_df = pd.read_pickle(path.join('pandas_data', 'aspect_category_detection_test.pkl'))

x_train, y_train = train_df.text, train_df.matrix
x_test, y_test = test_df.text, test_df.matrix


print(y_test)
text_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
     ('svc', SGDClassifier()),    
     ])
 
text_clf.fit(x_train, y_train)

predicted = text_clf.predict(x_test)
mean = np.mean(predicted == y_test)

print(text_clf)

# print('Test Mean: {}'.format(mean))
# print('---------------')
# print('Test Precision: {}'.format(precision_score(y_test, predicted, average="macro")))
# print('Test Recall: {}'.format(recall_score(y_test, predicted, average="macro")))
# print('---------------')
# print('Test F1: {}'.format(f1_score(y_test, predicted, average="macro")))
# print(roc_auc_score(y_test, predicted))
# print(confusion_matrix(y_test, predicted))

# Test Precision: 0.6985707077355097
# Test F1: 0.7122212245492272