import pandas as pd
import xml.etree.ElementTree as et
from collections import Counter
import re


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn import svm
import numpy as np 


XML_PATH = "/home/saif/uni/absa-diss/ABSA16_Laptops_Train_SB1_v2.xml"
TEST_XML_PATH = "/home/saif/absa-diss/ABSA16_Laptops_Test_SB1.xml"

tree = et.parse(XML_PATH)
reviews = tree.getroot()
sentences = reviews.findall('**/sentence')


opinions = reviews.findall('**/**/Opinion')
categories = [opinion.attrib['category'] for opinion in opinions]

def sentence_categories(xml_path = XML_PATH):
    """
        Returns lookup dictionary that has sentence id's categories
    """
    tree = et.parse(xml_path)
    reviews = tree.getroot()
    sentences = reviews.findall('**/sentence')

    sentence_categories = {}

    for sentence in sentences:
        for opinions in sentence:
            categories = []
            for opinion in opinions:
                categories.append(opinion.attrib['category'])

        sentence_categories[sentence.attrib['id']] = categories
    return sentence_categories



def pre_process_sentence(text):
    """
    Strips, removes
    """
    processed = re.sub(r'[^\w\s]','',df.text.lower())
    return(processed)

def df_sentences(xml_path = XML_PATH):
    """
        Takes XML Training data and returns a pandas dataframe of sentences;
        returns duplicate sentences if each sentence has multiple aspects of polarity   
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

        # except:
        #     polarity = 'None'
        #     category = 'None'
        #     sentences_list.append([sentence_id, sentence_text, category, polarity])

        except:
            pass

    return pd.DataFrame(sentences_list, columns = ["id", "text", "category", "polarity"])


def tokenizer(df):
    """
        Takes a sentence, and reuturns a matrix
    """

    df.text



def df_subjectivity(xml_path = XML_PATH):
    """
        Takes XML Training data and returns a pandas dataframe of unique sentences;
        with subjectivity 1 if they express an opinion, 0 if not 
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

            num_opinions = 0 
            count_subjectivity = 0
            for opinion in opinions:
                polarity = opinion.attrib['polarity']
                num_opinions+=1
                if polarity == "positive" or polarity == "negative":
                    count_subjectivity += 1
            
            if (count_subjectivity !=0):
                sentences_list.append([sentence_id, sentence_text, 1])
            else:
                sentences_list.append([sentence_id, sentence_text, 0])

        except:
            # Ignore sentences that do have any opinions labelled
            pass



    return pd.DataFrame(sentences_list, columns = ["id", "text", "subjectivity"])


def df_test_subjectivity(xml_path = XML_PATH):
    """
        Takes XML Test data and returns a pandas dataframe of sentences;
    """

    tree = et.parse(xml_path)
    reviews = tree.getroot()
    sentences = reviews.findall('**/sentence')

    sentences_list = []

    for sentence in sentences:

        sentence_id = sentence.attrib['id']                
        sentence_text = sentence.find('text').text
        sentences_list.append([sentence_id, sentence_text])



    return pd.DataFrame(sentences_list, columns = ["id", "text"])


def df_aspect(xml_path = XML_PATH):
    """
        Takes XML Training data and returns a pandas dataframe of sentences;
        returns duplicate sentences if each sentence has multiple aspects of polarity   
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
            polarity = 'None'
            category = 'None'
            sentences_list.append([sentence_id, sentence_text, category, polarity])


def df_polarity(xml_path = XML_PATH):
    """
        Takes XML Training data and returns a pandas dataframe of sentences;
        returns duplicate sentences if each sentence has multiple aspects of polarity   
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
                
                if(polarity == "positive"):
                    polarity_val = 1
                elif(polarity == "negative"):
                    polarity_val = 0 
                sentences_list.append([sentence_id, sentence_text, polarity_val])

        except Exception as se:
            # print('Uhoh, got SomeException:' + se.args[0])
            pass


    return pd.DataFrame(sentences_list, columns = ["id", "text", "polarity"])




def df_aspect_category(xml_path = XML_PATH):
    """
        Takes XML Training data and returns a pandas dataframe of sentences;
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

def assign_category(xml_path, n=16):
    """
        Returns dictionary of n most common categories 
    """

    sentences = df_aspect_category(XML_PATH)
    categories = Counter(sentences.category).most_common(n)

    common_categories = [category_tuple[0] for category_tuple in categories]

    common_df = sentences[sentences['category'].isin(common_categories)]

    assigned = {}

    for i in range(0, len(common_categories)):
        assigned[common_categories[i]] = i 

    return assigned
    # common_df[''] = common_df[]

    

    return None


    

def df_categories(xml_path, n=16):
    """
        Takes XML Training data and returns a pandas dataframe of sentences;
    """

    tree = et.parse(xml_path)
    reviews = tree.getroot()
    sentences = reviews.findall('**/sentence')

    sentences_list = []
    
    category_dict = assign_category(xml_path, n)

    for sentence in sentences:

        sentence_id = sentence.attrib['id']                
        sentence_text = sentence.find('text').text
        category_matrix = np.zeros((n, ), dtype=int)
        sentence_text =  re.sub(r'[^\w\s]','',sentence_text.lower())

        try: 
            opinions = list(sentence)[1]
            categories = []
            for opinion in opinions:
                categories.append(opinion.attrib['category'])
                location = category_dict[opinion.attrib['category']]
                category_matrix[location] = 1
                # category_matrix[i] = assigned(category_dict[opinion.attrib['category']])

            sentences_list.append([sentence_id, sentence_text, categories, category_matrix])

        except:
            pass

    return pd.DataFrame(sentences_list, columns = ["id", "text", "category", "matrix"])

XML_SB1_TEST_GOLD_PATH = "/home/saif/uni/absa-diss/EN_LAPT_SB1_TEST_.xml.gold"


df = df_categories(XML_SB1_TEST_GOLD_PATH)
print(df)
df.to_pickle('aspect_category_detection_gold_test.pkl')

# negative = df.loc[(df["polarity"] == 1)]
# print(negative)
# print(len(negative))



# category_dict = assign_category(XML_PATH, 16)
# print(category_dict)
# df = df_categories(XML_PATH, 16)
# print(df.matrix.to_numpy())
# df = df_categories(XML_PATH, 16)
# df.to_pickle('tensorflow_text.pkl')




# def make_categories_dictionary(df):

# df = df_sentences(XML_PATH)

# count = len(df[df.polarity == 'neutral'])
# print(count)

# # Categories
# sentences = df_sentences(XML_PATH)
# categories = Counter(sentences.category).most_common(10) 
# len_categories = len(categories) #82 Categories

# Categories
# sentences = df_sentences(XML_PATH)
# categories = Counter(sentences.category).most_common(10) 
# nonecat = sentences[sentences.category == "None"] 

# Categories

# ('LAPTOP#GENERAL', 634), 
# ('LAPTOP#OPERATION_PERFORMANCE', 278), 
# ('LAPTOP#DESIGN_FEATURES', 253), 
# ('LAPTOP#QUALITY', 224),
# ('LAPTOP#MISCELLANEOUS', 142),
# ('LAPTOP#USABILITY', 141), 
# ('SUPPORT#QUALITY', 138), 
# ('LAPTOP#PRICE', 136), 
# ('COMPANY#GENERAL', 90), 
# ('BATTERY#OPERATION_PERFORMANCE', 86), 
# ('LAPTOP#CONNECTIVITY', 55), 
# ('DISPLAY#QUALITY', 53), 
# ('LAPTOP#PORTABILITY', 51), 
# ('OS#GENERAL', 35), 
# ('SOFTWARE#GENERAL', 31), 
# ('KEYBOARD#DESIGN_FEATURES', 29)]


# assign_category(XML_PATH)

# sentences = df_sentences(XML_PATH)
# print(sentences)
# categories = Counter(sentences.category).most_common(16)
# # len_categories = len(categories) #82 Categories

# common_categories = [category_tuple[0] for category_tuple in categories]
# common_df = sentences[sentences['category'].isin(common_categories)]

# matrix =  np.zeros((n,), dtype=int)
# assigned = {}

# for i in range(0, len(common_categories)):
#     assigned[common_categories[i]] = i 

# print(assigned)











# df = df_subjectivity(XML_PATH)
# print(df)
# vectorizer = CountVectorizer()

## --------------------------------------------------


# from sklearn.pipeline import Pipeline
# from sklearn.linear_model import SGDClassifier
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.feature_extraction.text import TfidfTransformer
# from sklearn.metrics import confusion_matrix
# from sklearn.model_selection import train_test_split
# from sklearn.svm import SVC


# df = df_categories(XML_PATH, n=16)
# print(df)

# df.sort_values("matrix", inplace = True) 


# duplicateRowsDF = df[df.duplicated()]
# print("Duplicate Rows except first occurrence based on all columns are :")
# print(duplicateRowsDF)


# X_train, X_test, y_train, y_test = train_test_split(df.text, df.matrix, test_size = 0.3, random_state = 0)

# text_clf = Pipeline([
#     ('vect', CountVectorizer()),
#     ('tfidf', TfidfTransformer()),
#     ('clf', SVC(decision_function_shape='ovo'))   
#     ])


# print(y_train)




# print(y_train)
# text_clf.fit(X_train, y_train)
# dec = clf.decision_function([[1]])
# clf.decision_function_shape = "ovr"
# dec = clf.decision_function([[1]])
# dec.shape[1]

# predicted = text_clf.predict(X_test)

# mean = np.mean(predicted == y_test)
# print(mean)



## --------------------------------------------------
# common_df["category_index"] = sentences[sentences['category'].isin(common_categories)]

# print(common_df.head(5))
# print(categories)
# print(categories.values)

# for y in reversed(sorted(categories.values())):
#     k = popularity_data.keys()[popularity_data.values().index(y)]
#     print k + ':', y
#     del popularity_data[k]

# import matplotlib.pyplot as plt


# w = categories.Counter(l)
# plt.bar(w.keys(), w.values())



# plt.show()

# ----------------------------------------------
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

train_df = df_polarity(XML_PATH)
test_df = df_polarity(XML_SB1_TEST_GOLD_PATH)

# x_train, x_val, y_train, y_val = train_test_split(train_df.text, train_df.polarity, test_size = 0.3, random_state = 0)

x_train, y_train = train_df.text, train_df.polarity
x_test, y_test = test_df.text, test_df.polarity

text_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
     ('clf', SGDClassifier()),    
     ])

text_clf.fit(x_train, y_train)

predicted = text_clf.predict(x_test)
mean = np.mean(predicted == y_test)
print(mean)


# print(len(sentences))
# print(len(common_df))
# nonetype = sentences.loc[sentences['category'] == "None"]
# print(nonetype)

# from sklearn.pipeline import Pipeline
# from sklearn.linear_model import SGDClassifier
# from sklearn.feature_extraction.text import TfidfTransformer
# from sklearn.metrics import confusion_matrix
# from sklearn.model_selection import train_test_split

# df = df_subjectivity(XML_PATH)
# print(df)
# X_train, X_test, y_train, y_test = train_test_split(df.text[:5000], df.subjectivity[:5000], test_size = 0.3, random_state = 0)

# text_clf = Pipeline([
#     ('vect', CountVectorizer()),
#     ('tfidf', TfidfTransformer()),
#      ('clf', SGDClassifier()),    
#      ])

# text_clf.fit(X_train, y_train)
# predicted = text_clf.predict(X_test)

# mean = np.mean(predicted == y_test)
# print(mean)


# print(len(sentences))
# ----------------------------------------------------------------------
# from sklearn.pipeline import Pipeline
# from sklearn.linear_model import SGDClassifier
# from sklearn.feature_extraction.text import TfidfTransformer
# from sklearn.metrics import confusion_matrix
# from sklearn.model_selection import train_test_split

# df = df_subjectivity(XML_PATH)
# print(df)
# X_train, X_test, y_train, y_test = train_test_split(df.text[:5000], df.subjectivity[:5000], test_size = 0.3, random_state = 0)

# text_clf = Pipeline([
#     ('vect', CountVectorizer()),
#     ('tfidf', TfidfTransformer()),
#      ('clf', SGDClassifier()),    
#      ])

# text_clf.fit(X_train, y_train)
# predicted = text_clf.predict(X_test)

# mean = np.mean(predicted == y_test)
# print(mean)


# print(len(sentences))

# ---------------------------------------
# from sklearn.pipeline import Pipeline
# from sklearn.linear_model import SGDClassifier
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.feature_extraction.text import TfidfTransformer
# from sklearn.metrics import confusion_matrix
# from sklearn.model_selection import train_test_split

# df = df_polarity(XML_PATH)

# X_train, X_test, y_train, y_test = train_test_split(df.text, df.polarity, test_size = 0.3, random_state = 0)

# text_clf = Pipeline([
#     ('vect', CountVectorizer()),
#     ('tfidf', TfidfTransformer()),
#      ('clf', SGDClassifier(loss='hinge', penalty='l2', 
#      alpha=1e-3, random_state=42,
#      max_iter=5, tol=None)),    
#      ])

# text_clf.fit(X_train, y_train)
# predicted = text_clf.predict(X_test)

# mean = np.mean(predicted == y_test)
# print(mean)

# ----------------------------------------------------------------------

# df = df_subjectivity(XML_PATH)
# test_df = df_test_subjectivity(TEST_XML_PATH)

# from sklearn.pipeline import Pipeline
# from sklearn.linear_model import SGDClassifier
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.feature_extraction.text import TfidfTransformer



# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(df.text, df.subjectivity, test_size = 0.3, random_state = 0)

# text_clf = Pipeline([
#     ('vect', CountVectorizer()),
#     ('tfidf', TfidfTransformer()),
#      ('clf', SGDClassifier(loss='hinge', penalty='l2', 
#      alpha=1e-3, random_state=42,
#      max_iter=5, tol=None)),    
#      ])

# text_clf.fit(X_train, y_train)
# predicted = text_clf.predict(X_test)

# mean = np.mean(predicted == y_test)
# print(mean)

# ----------------------------------------------------------------------


































# vectorizer = CountVectorizer()
# X_train_counts = vectorizer.fit_transform(df.text)

# print(vectorizer.vocabulary_.get(u'algorithm'))
# clf = svm.SVC()(gamma='auto')
# clf.fit([train_df.vectorized, train_df.subjectivity], test_df.vectorized)

# docs_new = ['Laptop is alright, could be better, could be worse', 'This laptop screen gives me no opinions', "Love love love it"]
# X_new_counts = vectorizer.transform(docs_new)
# predicted = clf.predict(X_new_counts)




# SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
# SVM.fit(Train_X_Tfidf,Train_Y)

# np.mean(predicted == df.subjectivity)

# text_clf.fit(df.text, df.subjectivity)
# predicted = text_clf.predict(docs_new)

# print(df.dtypes)
# for index, row in df.iterrows():
#      print(row.text)

# df.to_pickle('test_subjectivity.pkl')

# categories = Counter(sentences.category).most_common(10) 

# print(df)
# count = len(df[df.subjectivity == 0])





# print("Count: " + str(count))
# print("Categories: " + str(categories))
# print(nonecat)


# def pd_subjectivity(xml_path = XML_PATH):
#     """
#         Takes XML training data and returns a pandas dataframe of sentences;
#         for use in subjectivity classification with positive and negative polarities
#         mapped to -1 or 1; with other values removed   
#     """

#     tree = et.parse(xml_path)
#     reviews = tree.getroot()
#     sentences = reviews.findall('**/sentence')

#     sentences_list = []

#     for sentence in sentences:

#         sentence_id = sentence.attrib['id']                
#         sentence_text = sentence.find('text').text

            
#         for opinion in opinions:
#             category = opinion.attrib['category']
#             polarity = opinion.attrib['polarity']
#             print(polarity)
#             if polarity == "positive":
#                 print(True)
#                 subjectivity = 1
#             elif polarity == "negative":
#                 subjectivity = -1
#             else: 
#                 subjectivity = None
#             if subjectivity != None:
#                 sentences_list.append([sentence_id, sentence_text, subjectivity])
    
#         return pd.DataFrame(sentences_list, columns = ["id", "text", "subjectivity"])

# def pd_subjectivity_helper(df):
#     if df['polarity'] == "negative":
#         val = -1
#     # elif df['polarity'] == "neutral":
#     #     val = 0
#     elif df['polarity'] == "positive":
#         val = 1
#     else:
#         val = 'None'
#     return val

# def pd_subjectivity(df):
#     """
#         Takes dataframe of xml from pd_sentences, adds subjectivity for positive and negative, 
#         deletes or None or neutral polarities
#     """

#     df['subjectivity'] = df.apply(pd_subjectivity_helper, axis=1)
#     del df['category']
#     del df['polarity']

#     df.dropna(axis=0, subset=['None'], inplace=True)

#     df = df.replace(to_replace=100).dropna()


#     return df


# subjectivity = pd_subjectivity(XML_PATH)
# print(subjectivity)
# print(pd_sentences(XML_PATH).head(10))


    # for review in reviews:
    #     print(review)
    #     rid = review.attrib["rid"]
    #     print(rid)

    # for review in reviews: 
    #     for sentences in reviews:
    #         print(review)
    