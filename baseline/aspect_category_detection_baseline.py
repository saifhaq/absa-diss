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

train_df = pd.read_pickle('aspect_category_detection_train_10_classes.pkl')
test_df = pd.read_pickle('aspect_category_detection_test_10_classes.pkl')

x_train, y_train = train_df.text, train_df.matrix
x_test, y_test = test_df.text, test_df.matrix

text_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
     ('svc', SGDClassifier()),    
     ])

text_clf.fit(x_train, y_train)

predicted = text_clf.predict(x_test)
mean = np.mean(predicted == y_test)

print('Test Mean: {}'.format(mean))
print('---------------')
print('Test Precision: {}'.format(precision_score(y_test, predicted, average="macro")))
print('Test Recall: {}'.format(recall_score(y_test, predicted, average="macro")))
print('---------------')
print('Test F1: {}'.format(f1_score(y_test, predicted, average="macro")))

# print(confusion_matrix(y_test, predicted))