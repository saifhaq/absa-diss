import xml.etree.ElementTree as et
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn import svm
import numpy as np 
import os.path as path
import re
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix, roc_auc_score, multilabel_confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np 
import os.path as path 
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick




subjectivity_df = pd.DataFrame(columns = ["Subjectivity", "Train Count", "Test Count", "Train Percentage", "Test Percentage"])

subjectivity_df = subjectivity_df.append({'Subjectivity': 'Subjective', 'Train Count': 0}, ignore_index=True)
subjectivity_df = subjectivity_df.append({'Subjectivity': 'Objective', 'Train Count': 0}, ignore_index=True)


# Calculate counts and percentages for train dataset 

train_df = pd.read_pickle(path.join('subjectivity', path.join('pandas_data', 'TRAIN_SUBJECTIVITY.pkl')))
flattened_subjectivities = train_df.subjectivity.to_list()
subjectivities_counter = Counter(flattened_subjectivities)
n_samples = sum(subjectivities_counter.values())

subjectivity_df.loc[0,'Train Count'] = subjectivities_counter[1]
subjectivity_df.loc[1,'Train Count'] = subjectivities_counter[0]

for index, row in subjectivity_df.iterrows():
    p = (row['Train Count'] / n_samples) *100
    
    row['Train Percentage'] = str('{0:.2f}'.format(p))


# # Calculate counts and percentages for test dataset 

test_df = pd.read_pickle(path.join('subjectivity', path.join('pandas_data', 'TEST_SUBJECTIVITY.pkl')))
flattened_subjectivities = test_df.subjectivity.to_list()
subjectivities_counter = Counter(flattened_subjectivities)
n_samples = sum(subjectivities_counter.values())

subjectivity_df.loc[0,'Test Count'] = subjectivities_counter[1]
subjectivity_df.loc[1,'Test Count'] = subjectivities_counter[0]

for index, row in subjectivity_df.iterrows():
    p = (row['Test Count'] / n_samples) *100
    
    row['Test Percentage'] = str('{0:.2f}'.format(p))


subjectivity_df.to_pickle(path.join('data_exploration', path.join('results', 'subjectivity_df.pkl')))

# # subjectivity_df = pd.read_pickle(path.join('data_exploration', path.join('results', 'subjectivity_df.pkl')))
train_percentages = [float(x) for x in subjectivity_df['Train Percentage']]
test_percentages = [float(x) for x in subjectivity_df['Test Percentage']]

print(subjectivity_df['Train Percentage'].to_list())


# create plot
n_groups = 2

fig, ax = plt.subplots()
# fig.set_size_inches(10, 10)

ax.yaxis.set_major_formatter(mtick.PercentFormatter())

index = np.arange(n_groups)

bar_width = 0.35
opacity = 0.8

def autolabel(rects):
    """
        Attach a text label above each bar in *rects*, displaying its height.
        https://matplotlib.org/3.1.1/gallery/lines_bars_and_markers/barchart.html#sphx-glr-gallery-lines-bars-and-markers-barchart-py
    """
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')




train = plt.bar(index + bar_width, train_percentages, bar_width,
alpha=opacity,
color='g',
label='Train')

test = plt.bar(index, test_percentages, bar_width,
alpha=opacity,
color='b',
label='Test')

autolabel(train)

autolabel(test)

plt.xlabel('Subjectivity')
plt.ylabel('Percentage of training samples')
plt.xticks(index + bar_width/2, ('Subjective', 'Objective'))
plt.legend()

plt.tight_layout()
plt.show()

print(subjectivity_df)