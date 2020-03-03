from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

embedding_layer = layers.Embedding(1000, 5)

result = embedding_layer(tf.constant([1,2,3]))
result.numpy()