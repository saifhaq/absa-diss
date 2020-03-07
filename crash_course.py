import tensorflow as tf 
from tensorflow import keras 
import numpy as np 
import matplotlib.pyplot as plt 

data = keras.data.imdb

(train_data, train_labels), (test_data, test_labels) = data.load_data(num_words=10000)
