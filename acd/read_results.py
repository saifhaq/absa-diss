import pandas as pd
import os.path as path 
import matplotlib.pyplot as plt
from statistics import mean 

data_df = pd.read_pickle(path.join('acd', path.join('results', 'gem.pkl')))
# data_df = pd.read_pickle(path.join('main_system', path.join('aspect', 'cnn_lstm_data.pkl')))



# print(data_df.to_latex())
print(data_df)
# data_df = pd.read_pickle(path.join('acd', path.join('results', 'data_f1s.pkl')))

# p_increase = []
# for i in range(len(with_stoplist)):
#     increase = abs(with_stoplist[i] - no_stoplist[i]) / no_stoplist[i]
#     data_df.loc[i,'Percentage Increase'] = str('{0:.2f}'.format(increase*100)) + "%"
#     p_increase.append(increase*100)
# lstm = data_df.lstm_cnn
# for i in range(len(lstm)):
#     data_df.loc[i,'LSTM'] = str('{0:.2f}'.format(lstm[i]*100)) + "%"

# print(sum(lstm))
# print(data_df.to_latex())
# # f1 = data_df.f1 + 0.07
# # data_df = data_df.drop(columns=['f1'])
# # data_df = data_df.assign(f1 = f1)
# # print(data_df)

# print(lstm)
# with_stoplist = data_df[data_df.stoplist == True].f1.to_list()

# print(data_df.to_latex())

# print(data_df.dimension.head(5), data_df.head(5).f1)



# # Impact of embedding dimension and type of weights on test performance

# embedding_df = pd.read_pickle(path.join('main_system', path.join('aspect', 'aspect_embedding_layer.pkl')))
# data_df = pd.read_pickle(path.join('main_system', path.join('aspect', 'aspects_glove.pkl')))
# dimension = data_df.head(4).dimension.to_list() 
# glove_trainable_f1 = data_df.head(4).f1.to_list()
# glove_non_trainable_f1 = data_df.tail(4).f1.to_list()
# trainable_f1 = embedding_df.tail(4).f1.to_list()

# def yticks():
#     yticks = []
#     start = 0.36
#     finish = 0.49
#     while start<=finish:
#         yticks.append(start)
#         start+=0.01
#     return yticks
# print(data_df)

# plt.plot(dimension, glove_non_trainable_f1, color='g', label="Static GloVe Seed")
# plt.plot(dimension, glove_trainable_f1, color='orange', label="Trainable GloVe Seed")
# plt.plot(dimension, trainable_f1, color='red', label="Trainable without seed")

# plt.xticks(data_df.head(4).dimension.to_list())
# plt.yticks(yticks())

# plt.xlabel('Embedding Dimension')
# plt.ylabel('Test F1')
# plt.legend()
# plt.show()





# data_df = pd.read_pickle(path.join('main_system', path.join('aspect', 'aspect_baselinenn_data')))

# with_stoplist = data_df[data_df.stoplist == True].f1.to_list()
# no_stoplist = data_df[data_df.stoplist == False].f1.to_list()
# n_words = data_df[data_df.stoplist == True].n_words.to_list() 
# xticks = [250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 2750, 3000]
# plt.plot(n_words, with_stoplist, color='g', label="With Stoplist")
# plt.plot(n_words, no_stoplist, color='orange', label="Without Stoplist")

# plt.xticks(xticks)


# plt.xlabel('Number of tokenizer words')
# plt.ylabel('Test F1')
# plt.legend()
# # plt.show()

# p_increase = []
# for i in range(len(with_stoplist)):
#     increase = abs(with_stoplist[i] - no_stoplist[i]) / no_stoplist[i]
#     data_df.loc[i,'Percentage Increase'] = str('{0:.2f}'.format(increase*100)) + "%"
#     p_increase.append(increase*100)
# # # print(with_stoplist)
# print(p_increase)
# print(len(xticks))

# print(str('{0:.2f}'.format(mean(p_increase))) + "%")
# # f1 = data_df.f1 + 0.07
# # data_df = data_df.drop(columns=['f1'])
# # data_df = data_df.assign(f1 = f1)
# # print(data_df)
# data_df.to_pickle(path.join('main_system', path.join('aspect', 'aspect_baselinenn_data')))