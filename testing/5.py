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

def stoplist(file_name = "stopwords.txt"):
    """
        Returns an array of stopwords, from each line of the 
        *file_name* text file
    """
    stopwords_txt = open(path.join('preprocessing', file_name))
    stoplist = []
    for line in stopwords_txt:
        values = line.split()
        stoplist.append(values[0])
    stopwords_txt.close()
    return stoplist

stopwords = stoplist()

df = pd.DataFrame(columns = ['text'])
text = "this sentence has some stopwords"
df = df.append({'text': text}, ignore_index=True)
df['text'] = df['text'].apply(lambda x: ' '.join([item for item in x.split() if item not in stopwords]))

print(df.text[0].split(" "))

