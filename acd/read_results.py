import pandas as pd
import os.path as path 
import matplotlib.pyplot as plt
from statistics import mean 

print("------------------------------------------------")
print("Aspect category detection F1-results")
data_f1s = pd.read_pickle(path.join('acd', path.join('results', 'data_f1s.pkl')))
print(data_f1s)
print("------------------------------------------------")

print("------------------------------------------------")
print("Aspect category detection per category percentage correctly identified")
data_df = pd.read_pickle(path.join('acd', path.join('results', 'data_df.pkl')))
print(data_df) 
print("------------------------------------------------")

print("------------------------------------------------")
print("Aspect category detection LSTM with tuning results")
tuner_df = pd.read_pickle(path.join('acd', path.join('results', 'acd_keras_tuner_results.pkl')))
tuner_results = tuner_df.head(10)
print(tuner_results)
print("------------------------------------------------")

