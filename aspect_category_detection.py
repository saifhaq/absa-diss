import tensorflow as tf
import pandas as pd 
from sklearn.feature_extraction.text import CountVectorizer
import string
import re
import numpy as np
from collections import Counter 
from itertools import dropwhile


# def pre_process_sentence(text):
#     """
#     Strips, removes
#     """

# def sequence_to_text(list_of_indices):
#     # Looking up words in dictionary
#     words = [tokenizer.index_word.get(letter) for letter in list_of_indices]
#     return(words)

df = pd.read_pickle('aspect_category_detection.pkl')
# results = set()

counter = Counter(df.text) 
# x = df['text'].str.lower().str.split().appy(counter)
x = Counter(" ".join(df.text).split(" "))

print(len(x))

# for key, count in dropwhile(lambda key_count: key_count[1] >= 200, 10):
#     del x[key]

# # most_occur = counter.most_common(100) 

# # x = Counter(results).most_common(10)
# print(x.most_common(10))

# cap_vector = tf.keras.preprocessing.sequence.pad_sequences(num_words=1000, train_seqs, padding='post')


# tf.keras.preprocessing.text.Tokenizer(
#     num_words=1000, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True,
#     split=' '
#     , char_level=False, oov_token=None, document_count=0, **kwargs
# )

