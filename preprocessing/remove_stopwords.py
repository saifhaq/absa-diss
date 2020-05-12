import os.path as path

sentence = "Hello what a wonderful world and this and and hello okay there"

file_name = "stopwords.txt"
stopwords_txt = open(path.join('preprocessing', file_name))
stoplist = []
for line in stopwords_txt:
    values = line.split()
    stoplist.append(values[0])
stopwords_txt.close()

x = ' '.join([item for item in sentence.split() if item not in stoplist])
print(x)