import matplotlib.pyplot as plt
import pickle
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

column_names = ['comprimento', 'diametro', 'altura', 'peso_inteiro', 'peso_sem_casca', 'peso_viceras', 'peso_concha', 'n_aneis']  # input
dataset = pd.read_csv('abalone.data', header=None, names=column_names,sep=",")
train_data = dataset[:3133]
data_X = train_data.iloc[:, 1:8]
data_Y = train_data.iloc[:, 8:9]

print(data_X)
print(data_Y)
regr = linear_model.LinearRegression()
preditor_linear_model = regr.fit(data_X, data_Y)
with open('abalone_model.pkl', 'wb') as preditor_Pickle:
    print("abalone.data")
    pickle.dump(preditor_linear_model, preditor_Pickle)