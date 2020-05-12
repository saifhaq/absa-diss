import pandas as pd
import os.path as path
import xml.etree.ElementTree as et
from collections import Counter

data_df = pd.DataFrame(columns = ["model", "acc", "f1"])

print(data_df)
data_df.to_pickle(path.join('polarity', path.join('results', 'data_df.pkl')))
