import matplotlib.pyplot as plt
import pickle as p1
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

column_names = ['comprimento', 'diametro', 'altura', 'peso_inteiro', 'peso_sem_casca', 'peso_viceras', 'peso_concha', 'n_aneis']#input
dataset = pd.read_csv('abalone.data', header=None, names=column_names)

train_data  =   dataset[:3133]
data_X=train_data.iloc[:,1:8]
data_Y=train_data.iloc[:,8:9]
#print(train_data.columns)
print(data_X)
print(data_Y)
regr = linear_model.LinearRegression()
preditor_linear_model = regr.fit(data_X, data_Y)
preditor_Pickle = open('abalone.data.pkl', 'wb')
print("abalone.data")
p1.dump(preditor_linear_model, preditor_Pickle)