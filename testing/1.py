import pandas as pd 
import os.path as path 
import re 


data_df = pd.DataFrame(columns = ['text'])

text = " Text: THIS IS A SENTENCE and needs to be normalized ??\#\#\£\%\^\&\*()!"
sentence_text =  re.sub(r'[^\w\s]','',text.lower())

data_df = data_df.append({'text': sentence_text}, ignore_index=True)

print(data_df)