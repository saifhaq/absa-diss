
import pandas as pd
import os.path as path 
import matplotlib.pyplot as plt
from statistics import mean 

embedding_df = pd.read_pickle(path.join('inputs', path.join('results', 'aspects_embedding.pkl')))
data_df = pd.read_pickle(path.join('inputs', path.join('results', 'aspects_glove.pkl')))

dimension = data_df.head(4).dimension.to_list() 
glove_trainable_f1 = data_df.head(4).f1.to_list()
glove_non_trainable_f1 = data_df.tail(4).f1.to_list()
trainable_f1 = embedding_df.tail(4).f1.to_list()

def yticks():
    yticks = []
    start = 0.36
    finish = 0.49
    while start<=finish:
        yticks.append(start)
        start+=0.01
    return yticks

plt.plot(dimension, glove_non_trainable_f1, color='g', label="Static GloVe Seed")
plt.plot(dimension, glove_trainable_f1, color='orange', label="Trainable GloVe Seed")
plt.plot(dimension, trainable_f1, color='red', label="Trainable without seed")

plt.xticks(data_df.head(4).dimension.to_list())
plt.yticks(yticks())

plt.xlabel('Embedding Dimension')
plt.ylabel('Test F1')
plt.legend()
plt.show()