import tensorflow as tf
import pandas as pd 
from tensorflow import keras
from tensorflow.keras import layers
import string
import re

# import sklearn.feature_extraction.text.CountVectorizer as CountVectorizer
# from sklearn import CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer



df = pd.read_pickle('subjectivity.pkl')
# print(df)
# print(df.dtypes)

# df = df[['text','subjectivity']]
# c = df.astype(str)

# df = df.apply(lambda text: text.str.strip())
# df = c.apply(lambda text: text.lower())

# vectorizer = CountVectorizer(stop_words='english')
# df["cunt"] = vectorizer.fit_transform(df.text).toarray()

# allDataVectorized = pd.DataFrame(vectorizer.fit_transform(df[['text']]))
# print(vectorizer())
# print(len(dt_mat.toarray()))

# tokens = []
# tokens.append(df.text)

# df.text = df.text.apply(str)

# df.text = df.text.str.replace('[^A-z ]','').str.replace(' +',' ').str.strip().lower()

# for sentence in df.text:

#     sentence = re.sub(r'[^\w\s]','',sentence)

#     print(sentence.lower().split(" "))


# print(vectorizer.get_feature_names())
# counts = vectorizer.transform(df.text)
# print(counts)

# text = re.sub(r'[^\w\s]','',df.text[0]).lower()
# text = text.split(" ")


vectorizer = CountVectorizer(stop_words="english")
X = vectorizer.fit_transform(df.text)


text = [df.text[0]]
print(text)

print(len(vectorizer.transform(text).toarray()[0]))


# print(df.text)
# print(df['text'].head(30))
# (train_data, test_data), info = tfds.load(
#     'imdb_reviews/subwords8k', 
#     split = (tfds.Split.TRAIN, tfds.Split.TEST), 
#     with_info=True, as_supervised=True)

# tf.keras.preprocessing.text.Tokenizer(
#     num_words=1000, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True,
#     split=' '
#     , char_level=False, oov_token=None, document_count=0, **kwargs
# )
# df = pd.DataFrame({'sentences': ['This is a very good site. I will recommend it to others.', 'Can you please give me a call at 9983938428. have issues with the listings.', 'good work! keep it up']})
# df['tokenized_sents'] = df.apply(lambda row: nltk.word_tokenize(row['sentences']), axis=1)
