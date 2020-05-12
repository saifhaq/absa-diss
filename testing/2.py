import pandas as pd 
import os.path as path 
import re 

train_df = pd.read_pickle(path.join('subjectivity', path.join('pandas_data', 'TRAIN_SUBJECTIVITY.pkl')))

test_df = pd.read_pickle(path.join('subjectivity', path.join('pandas_data', 'TEST_SUBJECTIVITY.pkl')))

print(len(train_df))
print(len(test_df))