import pandas as pd
import os.path as path
import xml.etree.ElementTree as et
from collections import Counter

data_f1s = pd.DataFrame(columns = ["model", "f1"])

print(data_f1s)
data_f1s.to_pickle(path.join('acd', path.join('results', 'data_f1s.pkl')))
