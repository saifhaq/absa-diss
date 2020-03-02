import pandas as pd
import xml.etree.ElementTree as et

XML_PATH = "/home/saif/uni/diss/absa-diss/ABSA16_Laptops_Train_SB1_v2.xml"

tree = et.parse(XML_PATH)

# bye = hi.attrib('sentences')

reviews = tree.getroot()
# print(reviews)
sentences = reviews.findall("**/sentence")
# print(sentences[10].attrib["id"])



opinions = reviews.findall("**/**/Opinion")
categories = [opinion.attrib["category"] for opinion in opinions]
aset = {}


def sentence_categories(xml_path = XML_PATH):
    """
        Returns lookup dictionary that has sentence id's categories
    """
    tree = et.parse(xml_path)
    reviews = tree.getroot()
    sentences = reviews.findall("**/sentence")

    sentence_categories = {}

    for sentence in sentences:
        for opinions in sentence:
            categories = []
            for opinion in opinions:
                categories.append(opinion.attrib["category"])

        sentence_categories[sentence.attrib["id"]] = categories
    return sentence_categories



def pre_process_sentence(text):
    """
    Strips, removes
    """

def pd_sentences(xml_path = XML_PATH):
    """
        Takes XML Training data and returns a pandas dataframe of sentences;
        returns duplicate sentences if each sentence has multiple aspects of polarity   
    """

    tree = et.parse(xml_path)
    reviews = tree.getroot()
    sentences = reviews.findall("**/sentence")

    # id_text_dict = {}
    sentences_list = []

    for sentence in sentences:
    
        sentence_id = sentence.attrib["id"]                
        sentence_text = sentence.find('text').text
        # id_text_dict[sentence.attrib["id"]] = text
 
        # print(str(sentence_id) + " " + str(len(opinions)))
        
        try: 
            opinions = sentence.getchildren()[1]

            for opinion in opinions:
                category = opinion.attrib["category"]
                polarity = opinion.attrib["polarity"]
                sentences_list.append([sentence_id, sentence_text, category, polarity])

        except:
            polarity = None
            category = None
            sentences_list.append([sentence_id, sentence_text, category, polarity])


            
        # for opinions in sentence:
        #     sentence_id = sentence_id
        #     if len(opinions):
        #         for opinion in opinions:
        #             category = opinion.attrib["category"]
        #             polarity = opinion.attrib["polarity"]
        #             sentences_list.append([sentence_id, sentence_text, category, polarity])
        #     else:
        #         polarity = None
        #         category = None
        #         sentences_list.append([sentence_id, sentence_text, category, polarity])
    return pd.DataFrame(sentences_list)
    # return sentences_list

# sentence_categories = sentence_categories(XML_PATH)
# print(len(sentence_categories.get("278:0")))
# print(pd_sentences(XML_PATH))
print(pd_sentences(XML_PATH).head(10))
# pd_sentences(XML_PATH)



# print(pd_sentences(XML_PATH))

# def pd_sentences(xml_path = XML_PATH):
#     """
#         Takes XML Training data and returns a pandas dataframe of sentences;
#         returns duplicate sentences if each sentence has multiple aspects of polarity   
#     """
#     for sentence in sentences:
#         sentence_id = sentence.attrib["id"]
#         text = ""
#         category = ""
#         polarity = ""
#     return True


# for sentence in sentences:
#     sentence_id = sentence.attrib["id"]
#     text = ""
#     category = ""
#     polarity = ""

#     print(sentence.findall('category'))
    
    # for opinion in opinions:
    #     categories = opinion.attrib["category"]
        
    

    # for categories in sentence.iter('opinions'):
    #     print(categories)

    # children = sentence.getchildren()

    # print(sentence.attrib[])
    # for opinions in sentence:
    #     for opinion in opinions:

    #         list_sentences.append([sentence_id,])

    #         aset[opinion.attrib["category"]] =   

    # for text in sentence:
    #     aset[sentence_id] = text.text
    #     print(text.text)

    # for opinions in sentence:
    #     for opinion in opinions:
    #         categories = opinion.attrib("category")
    # print(sentence.text)
# ods = sentences[0].attrib["isentence_text
"""
    Takes XML Training data and returns a pandas dataframe of sentences;
    returns duplicate sentences if each sentence has multiple aspects of polarity   
"""

# def pd_sentences(xml_path):
#     tree = et.parse()

#     xml_doc = et.parse(xml_path)
#     doc = xml_doc.getroot()
 
#     attrib = doc.attrib

#     for xml in xml_doc.iter('sentences'):
#         print(xml)
#         doc_dict = attrib.copy()
#         doc_dict.update(xml.attrib)
#         doc_dict['data'] = xml.text
#         yield doc_dict
    
#     # return True
    
#     return pd.DataFrame(list(doc_dict))


# sentences = pd_sentences("/home/saif/uni/aspect-based-sentiment-analysis/ABSA16_Laptops_Train_SB1_v2.xml")


# def pd_sentences(xml_path):
    
#     xml_doc = et.parse(xml_path)
#     doc = xml_doc.getroot()
 
#     attrib = doc.attrib

#     for xml in doc.iter('sentences'):
#         doc_dict = attrib.copy()
#         doc_dict.update(xml.attrib)
#         doc_dict['data'] = xml.text
#         # yield doc_dict
    
#     # return True
    
#     return pd.DataFrame(list(et.getroot(xml_path)))

# sentences = pd_sentences("/home/saif/uni/aspect-based-sentiment-analysis/ABSA16_Laptops_Train_SB1_v2.xml")
# print(sentences.iloc[0, :])

# def pd_sentences(xml_path):
    
#     xml_doc = et.parse(xml_path)
#     doc = xml_doc.getroot()
 
#     attrib = doc.attrib

#     for xml in doc.iter('sentences'):
#         doc_dict = attrib.copy()
#         doc_dict.update(xml.attrib)
#         doc_dict['data'] = xml.text
#         # yield doc_dict
    
#     # return True
    
#     return pd.DataFrame(list(et.getroot(xml_path)))

# def intr_docs(xml_doc):
#     attr = xml_doc.attrib 

#     for xml in xml_doc.iter('sentences'):
#         doc_dict = attr.copy()
#         doc_dict.update(xml.attrib)
#         doc_dict['data'] = xml.text

#         yield doc_dict
        
# etree = et.parse("/home/saif/uni/aspect-based-sentiment-analysis/ABSA16_Laptops_Train_SB1_v2.xml")
# doc_df = pd.DataFrame(list(intr_docs(etree.getroot()))) 


# print(doc_df.iloc[0, :])
        # sentences = pd_sentences("/home/saif/uni/aspect-based-sentiment-analysis/ABSA16_Laptops_Train_SB1_v2.xml")
# print(sentences.iloc[0, :])