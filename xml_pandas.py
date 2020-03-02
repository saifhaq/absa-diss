import pandas as pd
import xml.etree.ElementTree as et

tree = et.parse("/home/saif/uni/aspect-based-sentiment-analysis/ABSA16_Laptops_Train_SB1_v2.xml")
reviews = tree.getroot()
sentences = reviews.findall("**/sentence")
ods = sentences[0].attrib["id"]

df = pd.DataFrame(list())

print(ods)

"""
    Takes XML Training data and returns a pandas dataframe of sentences;
    returns duplicate sentences if each sentence has multiple categories  
"""
def pd_sentences(xml_path):
    attribute = xml_doc.attrib
    
    for xml in xml_path('sentences'):
        doc_dict = attrib.copy()
        docs_dict.update(xml.attrib)
        doc_dict['data'] = xml.text
        yield doc_dict
        
    return pd.DataFrame(list(et.parse(xml_path)))

sentences = pd_sentences("/home/saif/uni/aspect-based-sentiment-analysis/ABSA16_Laptops_Train_SB1_v2.xml")
print(sentences)