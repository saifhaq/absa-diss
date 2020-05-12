import pandas as pd 
import os.path as path 
import re 

train_df = pd.read_pickle(path.join('polarity', path.join('pandas_data', 'TRAIN_POLARITY.pkl')))
test_df = pd.read_pickle(path.join('polarity', path.join('pandas_data', 'TEST_POLARITY.pkl')))

print(len(train_df))
print(len(test_df))
