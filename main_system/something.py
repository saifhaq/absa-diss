import pandas as pd 
import os.path as path 

# train_df = pd.read_pickle(path.join('pandas_data', 'aspect_category_detection_train_'+str(16)+'_classes.pkl'))
df = pd.read_pickle(path.join('baseline', path.join('aspect', 'aspect_baseline_data')))

# print(df.text
# )
df.at[0,'desired_category'] = 'something'
# print(df.at[0,'desired_category'])
print(df)