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

def df_aspect_category(xml_path):
    """
        Takes *xml_path* and returns dataframe of each sentence and corresponding category. 
        If sentence has multiple categories, the sentence is returned multiple times. 
        
        Dataframe returned as: [id, text, category, polarity] 
    """

    tree = et.parse(xml_path)
    reviews = tree.getroot()
    sentences = reviews.findall('**/sentence')

    sentences_list = []

    for sentence in sentences:

        sentence_id = sentence.attrib['id']                
        sentence_text = sentence.find('text').text

        try: 
            opinions = list(sentence)[1]

            for opinion in opinions:
                category = opinion.attrib['category']
                polarity = opinion.attrib['polarity']
                sentences_list.append([sentence_id, sentence_text, category, polarity])

        except:
            pass

    return pd.DataFrame(sentences_list, columns = ["id", "text", "category", "polarity"])

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


n = 16
data_df = pd.DataFrame(columns = ["Aspect Category", "Train Samples", "Test Samples", "Train Percentage", "Test Percentage"])
category_dict = {'LAPTOP#GENERAL': 0, 'LAPTOP#OPERATION_PERFORMANCE': 1, 'LAPTOP#DESIGN_FEATURES': 2, 'LAPTOP#QUALITY': 3, 'LAPTOP#MISCELLANEOUS': 4, 'LAPTOP#USABILITY': 5, 'SUPPORT#QUALITY': 6, 'LAPTOP#PRICE': 7, 'COMPANY#GENERAL': 8, 'BATTERY#OPERATION_PERFORMANCE': 9, 'LAPTOP#CONNECTIVITY': 10, 'DISPLAY#QUALITY': 11, 'LAPTOP#PORTABILITY': 12, 'OS#GENERAL': 13, 'SOFTWARE#GENERAL': 14, 'KEYBOARD#DESIGN_FEATURES': 15}

TRAIN_XML_PATH = "ABSA16_Laptops_Train_SB1_v2.xml"
sentences_train = df_aspect_category(TRAIN_XML_PATH)
categories_train = Counter(sentences_train.category).most_common(n)

TEST_XML_PATH = "ABSA16_Laptops_Test_GOLD_SB1.xml"
sentences_test = df_aspect_category(TEST_XML_PATH)
categories_test = Counter(sentences_test.category).most_common(n)

    
for i in range(0, n):
    DESIRED_CATEGORY = categories_train[i][0]
    TRAIN_COUNT = categories_train[i][1]
    TEST_COUNT = categories_test[i][1]
    data_df = data_df.append({'Aspect Category': DESIRED_CATEGORY, 'Train Samples': TRAIN_COUNT, 'Test Samples': TEST_COUNT}, ignore_index=True)

n_sentences_train = len(sentences_train)
n_sentences_test = len(sentences_test)

for index, row in data_df.iterrows():
    train = (row['Train Samples'] / n_sentences_train) *100
    row['Train Percentage'] = str('{0:.2f}'.format(train)) 

    test = (row['Test Samples'] / n_sentences_test) *100
    row['Test Percentage'] = str('{0:.2f}'.format(test)) 




# print(data_df)

# print(data_df.to_latex())



categories = [str(x) for x in data_df['Aspect Category']]
truncated_categories = []
for i in range(0, len(categories)):
    # print(len(c))
    chars = len(categories[i])
    if chars>=10:
        info = '(' + str(i) + ') '+categories[i][:10] + '..'
    else:
        info = categories[i]
    truncated_categories.append(info)

train_percentages = [float(x) for x in data_df['Train Percentage']]
test_percentages = [float(x) for x in data_df['Test Percentage']]


# # create plot
n_groups = 16

fig, ax = plt.subplots()
# fig.set_size_inches(10, 10)

ax.yaxis.set_major_formatter(mtick.PercentFormatter())

index = np.arange(n_groups)

bar_width = 0.4
opacity = 0.8

bars = []
for i in range(0,16):
    bars.append(plt.bar(index + bar_width, train_percentages, bar_width,
    alpha=opacity,
    color='g',
    ))
    autolabel(bars[0])


plt.xlabel('Category')
plt.ylabel('Percentage of total training samples')
plt.yticks([0, 5, 10, 15, 20, 25, 30])

plt.xticks(index + bar_width/2, (truncated_categories))
plt.xticks(rotation='vertical')

# plt.legend()

plt.tight_layout()
plt.show()

# data_df = data_df.drop(columns=['Test Percentage', 'Test Samples'])


# print(data_df.to_latex())
# data_df.to_pickle(path.join('data_exploration', path.join('results', 'category_distibutions.pkl')))

print(data_df)

# print(sentences_test)