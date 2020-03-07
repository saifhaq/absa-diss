# import tensorflow as tf
import pandas as pd 
import tensorflow as tf
import tensorflow_datasets as tfds
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

def plot_graphs(history, metric):
  plt.plot(history.history[metric])
  plt.plot(history.history['val_'+metric], '')
  plt.xlabel("Epochs")
  plt.ylabel(metric)
  plt.legend([metric, 'val_'+metric])
  plt.show()

# import os
import re


# df = pd.read_pickle('subjectivity.pkl')

df = pd.read_pickle('tensorflow_text.pkl')

# encoder = tfds.features.text.SubwordTextEncoder(
#     vocab_list=df.text
# )

# tf.keras.preprocessing.text.Tokenizer(
#     num_words=None, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True,
#     split=' ', char_level=False, oov_token=None, document_count=0,
# )

# https://www.tensorflow.org/tutorials/text/image_captioning
def calc_max_length(tensor):
    return max(len(t) for t in tensor)

category_dict = {'LAPTOP#GENERAL': 0, 'LAPTOP#OPERATION_PERFORMANCE': 1, 'LAPTOP#DESIGN_FEATURES': 2, 'LAPTOP#QUALITY': 3, 'LAPTOP#MISCELLANEOUS': 4, 'LAPTOP#USABILITY': 5, 'SUPPORT#QUALITY': 6, 'LAPTOP#PRICE': 7, 'COMPANY#GENERAL': 8, 'BATTERY#OPERATION_PERFORMANCE': 9, 'LAPTOP#CONNECTIVITY': 10, 'DISPLAY#QUALITY': 11, 'LAPTOP#PORTABILITY': 12, 'OS#GENERAL': 13, 'SOFTWARE#GENERAL': 14, 'KEYBOARD#DESIGN_FEATURES': 15}
category_dict_inverted = {v: k for k, v in category_dict.items()}


top_k = 5000
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=top_k,
                                                  oov_token="<unk>",
                                                  filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
tokenizer.fit_on_texts(df.text)


tokenizer.word_index['<pad>'] = 0
tokenizer.index_word[0] = '<pad>'

tokenizer.word_index['<unk>'] = 1
tokenizer.index_word[1] = '<unk>'


train_seqs = tokenizer.texts_to_sequences(df.text)
cap_vector = tf.keras.preprocessing.sequence.pad_sequences(train_seqs, padding='post')
max_length = calc_max_length(train_seqs)

# reverse_word_map = dict(map(reversed, tokenizer.word_index.items()))

def sequence_to_text(list_of_indices):
    # Looking up words in dictionary
    words = [tokenizer.index_word.get(letter) for letter in list_of_indices]
    return(words)

# img_name_train, img_name_val, cap_train, cap_val = train_test_split(cap_vector,
#                                                                     df.matrix,
#                                                                     test_size=0.2,
#                                                                     random_state=0)

# labels = pd.DataFrame(df['matrix'].tolist())
# l_1d = df.matrix.tolist()
# print(l_1d)
# df.matrix = df['matrix'].tolist()
# l = tf.convert_to_tensor(df.matrix)

some = df.matrix.to_numpy()
# l = [l.tolist() for l in some]
l = np.stack( some, axis=0 )

X_train, X_test, y_train, y_test = train_test_split(cap_vector, l, test_size = 0.2, random_state = 0)

print(X_train.shape)
print(y_train.shape)

print(sequence_to_text(X_train[0]))
print(X_train[0])

# print(df.matrix.reshape)
# print(df.matrix.shape)
# print((img_name_train).shape)
# print((cap_train).shape)
# print((img_name_val).shape)
# print((cap_val).shape)

# print(X_train.shape)
# print(y_train.shape)
# print(X_test.shape)
# print(y_test.shape)

# print("Test set (images) shape: {shape}".format(shape=data.test.images.shape))

# (train_data, test_data), info = tfds.load(
#     # Use the version pre-encoded with an ~8k vocabulary.
#     'imdb_reviews/subwords8k', 
#     # Return the train/test datasets as a tuple.
#     split = (tfds.Split.TRAIN, tfds.Split.TEST),
#     # Return (example, label) pairs from the dataset (instead of a dictionary).
#     as_supervised=True,
#     # Also return the `info` structure. 
#     with_info=True)


# features, labels = (np.random.sample((1284,73)), np.random.sample((1284,16)))
# print(features.shape)
# print(labels.shape)
# print("-----")
# print(X_train.shape)
# print(y_train.shape)
# dataset = tf.data.Dataset.from_tensor_slices((features,labels))


# print(X_test[0])
# print(y_train[0])


# features, labels = (X_train, y_train)
# train_dataset = tf.data.Dataset.from_tensor_slices((features,labels))

# features, labels = (X_test, y_test)
# test_dataset = tf.data.Dataset.from_tensor_slices((features,labels))


# for element in train_dataset: 
#   print(element) 


# model = tf.keras.Sequential([
#     tf.keras.layers.Embedding(vocab_size, 64),
#     tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
#     tf.keras.layers.Dense(64, activation='relu'),
#     tf.keras.layers.Dense(16)
# ])

# model.compile(optimizer='adam',
#               loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
#               metrics=['accuracy'])


# print(train_dataset)
# print(X_train.shape)
# tf_dataset_x_train = tf.data.Dataset.from_tensor_slices(X_train)
# tf_dataset_y_train = tf.data.Dataset.from_tensor_slices(y_train)

# print(tf_dataset_x_train)
# print(tf_dataset_y_train)

# train_dataset = tf.constant((X_train,y_train))
# test_dataset = tf.constant((X_test,y_test))

# train_dataset = tf.data.Dataset.from_tensor_slices((X_train,y_train))
# test_dataset = tf.data.Dataset.from_tensor_slices((X_test,y_test))

# BATCH_SIZE = 64
# BUFFER_SIZE = 1000
# embedding_dim = 256
# units = 512
# vocab_size = top_k + 1
# num_steps = len(img_name_train) // BATCH_SIZE
# # Shape of the vector extracted from InceptionV3 is (64, 2048)
# # These two variables represent that vector shape
# features_shape = 2048
# attention_features_shape = 64


# dataset = tf.data.Dataset.from_tensor_slices((img_name_train, cap_train))
# # Shuffle and batch
# dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
# dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

# val_dataset = tf.data.Dataset.from_tensor_slices((img_name_val, cap_val))
# # Shuffle and batch
# val_dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
# val_dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)




# embedding_layer = layers.Embedding(1000, 5)
# embedding_dim=16
# num_categories = 16


# model = tf.keras.Sequential([
#   layers.Embedding(vocab_size, embedding_dim),
#   layers.GlobalAveragePooling1D(),
#   layers.Dense(16, activation='relu'),
# #   layers.Dense(1)
# ])





# # print(val_dataset)
# # print(img_name_val[0])

# history = model.fit(
#     train_dataset,
#     epochs=10,
#     validation_data=test_dataset, validation_steps=20)


# plt.plot(history.history['accuracy'], label='accuracy')
# plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.ylim([0.5, 1])
# plt.legend(loc='lower right')

# test_loss, test_acc = model.evaluate(img_name_val,  cap_val, verbose=2)

# img_name_train, img_name_val, cap_train, cap_val = train_test_split(img_name_vector,
#                                                                     cap_vector,
#                                                                     test_size=0.2,
#                                                                     random_state=0)

# encoder = tfds.features.text.SubwordTextEncoder.build_from_corpus(
#         vocab_list=df.text
# )

# sample_string = 'Hello TensorFlow.'
# encoded_string = encoder.encode(sample_string)

# print(encoder.subwords)

# print(encoder.vocab_size)

# text = lambda: re.sub(r'[^\w\s]','',str(df.text).lower())
# df.apply(text)
# # text = re.sub(r'[^\w\s]','',df.text[1]).lower()
# # text = text.split(" ")

# # df.text = text
# # df.text = re.sub(r'[^\w\s]','',df.text.lower().split(" "))
# print(text)


# encoder = TFencoder(df.features['text'])
# print('Vocabulary size: {}'.format(encoder.vocab_size))

# def labeler(example, index):
#   return example, tf.cast(index, tf.int64)  

# labeled_data_sets = []

# for i, file_name in enumerate(FILE_NAMES):
#   lines_dataset = tf.data.TextLineDataset(os.path.join(parent_dir, file_name))
#   labeled_dataset = lines_dataset.map(lambda ex: labeler(ex, i))
#   labeled_data_sets.append(labeled_dataset)

# (train_data, test_data), info = tfds.load(
#     'imdb_reviews/subwords8k', 
#     split = (tfds.Split.TRAIN, tfds.Split.TEST), 
#     with_info=True, as_supervised=True)

#  encoder = info.features['text'].encoder
# encoder.subwords[:20]