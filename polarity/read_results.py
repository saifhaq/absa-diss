import pandas as pd
import os.path as path 
import matplotlib.pyplot as plt
import numpy as np 

data_df = pd.read_pickle(path.join('polarity', path.join('results', 'data_df.pkl')))
print(data_df)
# print(data_df.to_latex())