import tensorflow as tf
import pandas as pd 
from sklearn.feature_extraction.text import CountVectorizer
import string
import re


df = pd.read_pickle('subjectivity.pkl')
print(df)