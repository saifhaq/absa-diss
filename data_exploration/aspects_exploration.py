import tensorflow as tf
import pandas as pd 
from sklearn.feature_extraction.text import CountVectorizer
import string
import re
import numpy as np
from collections import Counter 
from sklearn.model_selection import train_test_split
import os
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import os.path as path

train_df = pd.read_pickle(path.join('pandas_data', 'aspect_category_detection_train.pkl'))
