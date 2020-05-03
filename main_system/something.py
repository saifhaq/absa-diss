import pandas as pd 
import os.path as path 
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

# train_df = pd.read_pickle(path.join('pandas_data', 'aspect_category_detection_train_'+str(16)+'_classes.pkl'))
# df = pd.read_pickle(path.join('baseline', path.join('aspect', 'aspect_embedding_layer.pkl')))
df = pd.read_pickle(path.join('main_system', path.join('aspect', 'aspects_glove.pkl')))
# data_df2 = pd.read_pickle(path.join('main_system', path.join('aspect', 'aspect_embedding_layer.pkl')))


# test_df = pd.read_pickle(path.join('pandas_data','aspect_category_detection_test_'+str(16)+'_classes.pkl'))


# print(df.text
# )
# df.at[0,'desired_category'] = 'something'
# print(df.at[0,'desired_category'])

# DESIRED_CATEGORY = 'LAPTOP#GENERAL'
# test_df_name = 'TEST.'+DESIRED_CATEGORY + '.pkl'


# df = pd.read_pickle(path.join('pandas_data', path.join('aspect_baseline', test_df_name)))


# test_loss, test_acc, test_precision, test_recall = models[0].evaluate(x_test, y_test)


# df = data_df.append(data_df)
# df = df.sort_values(by=['f1'])
print(df)
# print(data_df2)