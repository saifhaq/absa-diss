import pandas as pd
import xml.etree.ElementTree as et
from collections import Counter

XML_PATH = "/home/saif/uni/diss/absa-diss/ABSA16_Laptops_Train_SB1_v2.xml"

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

        except:
            polarity = 'None'
            category = 'None'
            sentences_list.append([sentence_id, sentence_text, category, polarity])



    return pd.DataFrame(sentences_list, columns = ["id", "text", "category", "polarity"])


# def tokenizer(sentence):
#     """
#         Takes a sentence, and reuturns a matrix
#     """

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
df = df_subjectivity(XML_PATH)
print(df.dtypes)
# for index, row in df.iterrows():
#      print(row.text)

# df.to_pickle('subjectivity.pkl')

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
    