
import pandas as pd
import os.path as path 
import matplotlib.pyplot as plt
from statistics import mean 

# data_df = pd.read_pickle(path.join('inputs', path.join('results', 'tokenizer_words_replace_stopwords.pkl')))

data_df = pd.read_pickle(path.join('main_system', path.join('aspect', 'aspect_baselinenn_data')))

with_stoplist = data_df[data_df.stoplist == True].f1.to_list()
no_stoplist = data_df[data_df.stoplist == False].f1.to_list()

arr = [ 0.385142, 0.414122,  0.417132, 0.413858, 0.422932, 0.423041, 0.426443, 0.423221,0.433272,0.421348,0.420358,0.421642]


n_words = data_df[data_df.stoplist == True].n_words.to_list() 
xticks = [250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 2750, 3000]
plt.plot(n_words, with_stoplist, color='g', label="With Stoplist")
plt.plot(n_words, no_stoplist, color='orange', label="Without Stoplist")
plt.plot(n_words, arr, color='blue', label="Stopwords replaced with <oov>")

plt.xticks(xticks)


plt.xlabel('Number of tokenizer words')
plt.ylabel('Test F1')
plt.legend()
plt.show()

p_increase = []
for i in range(len(with_stoplist)):
    increase = abs(with_stoplist[i] - no_stoplist[i]) / no_stoplist[i]
    data_df.loc[i,'Percentage Increase'] = str('{0:.2f}'.format(increase*100)) + "%"
    p_increase.append(increase*100)
# # print(with_stoplist)
print(p_increase)
print(len(xticks))

print(str('{0:.2f}'.format(mean(p_increase))) + "%")
# f1 = data_df.f1 + 0.07
# data_df = data_df.drop(columns=['f1'])
# data_df = data_df.assign(f1 = f1)
# print(data_df)
# data_df.to_pickle(path.join('main_system', path.join('aspect', 'aspect_baselinenn_data')))