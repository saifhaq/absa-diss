import pandas as pd
import os.path as path 
import matplotlib.pyplot as plt
import numpy as np 

data_df = pd.read_pickle(path.join('polarity', path.join('results', 'data_df.pkl')))
print(data_df)
# train_df = pd.read_pickle(path.join('polarity', path.join('pandas_data', 'TRAIN_POLARITY.pkl')))

# polarity_matrix = np.zeros((16, ), dtype=int)
# p = polarity_matrix[0] = -1


# matrix = train_df.polarity_matrix
# print(train_df.head(50))
# np.array_equal(matrix, p)

# train_df = train_df.loc[]

# print(train_df)

