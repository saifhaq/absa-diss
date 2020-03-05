import tensorflow as tf
import numpy as np
import pprint
import re


text = "Quo usque tandem abutere, Catilina, patientia nostra? Quamdiu etiam furor iste tuus nos eludet? Quem ad finem sese effrenata iactabit audacia?"

tokenized = re.sub('[,?.]','', text).lower().split(' ') #Let's tokenize our text by just take each word
vocab = {k:v for v,k in enumerate(np.unique(tokenized))}


EMBED_SIZE = 50
VOCAB_LEN = len(vocab.keys())

print(VOCAB_LEN)
# words_ids = tf.constant([vocab["abutere"], vocab["patientia"]])
