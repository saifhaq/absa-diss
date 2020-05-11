import pandas as pd
import os.path as path 
import matplotlib.pyplot as plt
from statistics import mean 

data_df = pd.read_pickle(path.join('acd', path.join('results', 'gem.pkl')))


# data_df = pd.read_pickle(path.join('acd', path.join('results', 'data_f1s.pkl')))
# data_df = pd.read_pickle(path.join('acd', path.join('results', 'data_df.pkl')))
# print(pd.read_pickle(path.join('acd', path.join('results', 'data_testing_lstm.pkl'))))

print(data_df)
# print(data_df)
# data_df = pd.read_pickle(path.join('main_system', path.join('aspect', 'cnn_lstm_data.pkl')))



# print(data_df.to_latex())
# print(data_df)
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
