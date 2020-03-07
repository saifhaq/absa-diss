# import tensorflow as tf
import pandas as pd 
# from tensorflow import keras
# from tensorflow.keras import layers
import string
import re

# import sklearn.feature_extraction.text.CountVectorizer as CountVectorizer
# from sklearn import CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.feature_extraction.text import TfidfTransformer

test_df = pd.read_pickle('test_subjectivity.pkl')
df = pd.read_pickle('subjectivity.pkl')
df = pd.read_pickle('tensorflow_text.pkl')


# train_df.text = train_df.text.str.replace('[^A-z ]','').str.replace(' +',' ').str.strip()
# train_df.text = re.sub(r'[^\w\s]','',train_df.text.lower().split(" ")
# X_train, X_test, y_train, y_test = train_test_split(train_df.text, train_df.subjectivity, test_size = 0.3, random_state = 0)

print(df)
vectorizer = CountVectorizer()
text_counts = vectorizer.fit_transform(df.text)

tfidf_transformer = TfidfTransformer()
text_idf = tfidf_transformer.fit_transform(text_counts)

print(text_idf[0])
# print(X_train_tfidf.shape)

# docs_new = ['Great laptop would highly recommend', 'OpenGL on the GPU is fast']
# X_new_counts = vectorizer.transform(docs_new)
# X_new_tfidf = tfidf_transformer.transform(X_new_counts)
# print(X_new_tfidf.toarray()[0])

# for i in range(0, len(X_new_tfidf.toarray()[0])):
#    print(X_new_tfidf.toarray()[0][i])

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

# train_df.text = train_df.text.str.replace('[^A-z ]','').str.replace(' +',' ').str.strip()
# train_df.text = re.sub(r'[^\w\s]','',train_df.text.lower().split(" ")

# for sentence in train_df.text:

#     sentence = re.sub(r'[^\w\s]','',sentence)

#     print(sentence.lower().split(" "))

# train_df.text = re.sub(r'[^\w\s]','',train_df.text) 

# train_df.text = train_df.text.apply(re.sub(r'[^\w\s]','',train_df.text) )

# print(vectorizer.get_feature_names())
# counts = vectorizer.transform(df.text)
# print(counts)

# text = re.sub(r'[^\w\s]','',df.text[0]).lower()
# text = text.split(" ")


# def tokenize_text(row):
#    return vectorizer.transform([row['text']]).toarray()

# vectorizer = CountVectorizer(stop_words="english", max_features=1000)
# X = vectorizer.fit_transform(train_df.text)


# df['vectorized'] = df.apply(tokenize_text, axis=1)
# train_df['vectorized'] = vectorizer.transform(train_df.text)
# test_df['vectorized'] = vectorizer.transform(test_df.text)

# print(df.vectorized[0][0])
# print(train_df.vectorized[0])


# x = train_df.vectorized
# y = train_df.subjectivity


# print(x)
# print(train_df)
# clf = svm.SVC()(gamma='auto')
# clf.fit([train_df.vectorized, train_df.subjectivity], test_df.vectorized)
# # svm.SVC(gamma='auto')

# sentence = "this sentence is straight facts"
# tokenized_sentence = vectorizer.transform([sentence])
# print(clf.predict(tokenized_sentence))

# text = [df.text[0]]
# print(text)

# print(len(vectorizer.transform(text).toarray()[0]))


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
