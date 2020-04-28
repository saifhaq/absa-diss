import pandas as pd 
import os.path as path 

# train_df = pd.read_pickle(path.join('pandas_data', 'aspect_category_detection_train_'+str(16)+'_classes.pkl'))
# df = pd.read_pickle(path.join('baseline', path.join('aspect', 'aspect_embedding_layer.pkl')))
data_df = pd.read_pickle(path.join('main_system', path.join('aspect', 'aspects_glove.pkl')))
data_df2 = pd.read_pickle(path.join('main_system', path.join('aspect', 'aspect_embedding_layer.pkl')))


# print(df.text
# )
# df.at[0,'desired_category'] = 'something'
# print(df.at[0,'desired_category'])
df = data_df.append(data_df)
df = df.sort_values(by=['f1'])
print(df)
# print(data_df2)