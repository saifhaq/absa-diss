import pandas as pd 
import os.path as path 


def stoplist(file_name = "stopwords.txt"):
  stopwords_txt = open(path.join('preprocessing', file_name))
  stoplist = []
  for line in stopwords_txt:
      values = line.split()
      stoplist.append(values[0])
  stopwords_txt.close()
  return stoplist

def apply_stoplist(df):
    stopwords = stoplist()
    for index, row in df.iterrows():
        split = row.text.split()
        sentence_words = []
        for item in split:
            if item in stopwords:
                item = '<oov>' 
            sentence_words.append(item)
        row.text = sentence_words
    return df

data_df = pd.DataFrame(columns = ['text'])
text = "this sentence has some stopwords"
data_df = data_df.append({'text': text}, ignore_index=True)
data_df = apply_stoplist(data_df)
print(data_df.text[0])