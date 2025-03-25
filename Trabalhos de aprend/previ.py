import matplotlib.pyplot as plt
import pickle as p1
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

data_x = input("Introduza valores do abalone (separados por ';'): \n")
data = data_x.split(";")
print(data)
fmap_data = map(float, data)
flist_data = np.array([list(fmap_data)])
print(flist_data)
data1 = pd.read_csv("abalone.data", sep=",")
data1_y = data1.iloc[:, 8]
data_preparation = pd.DataFrame(flist_data, columns=data1.columns[:8])
loaded_model = p1.load(open('abalone_model.pkl', 'rb'))
y_pred = loaded_model.predict(flist_data)
print("Predição da qualidade do abalone:", int(y_pred))
if int(y_pred) == data1_y.iloc[0]:
    print("abalone.data(X) =", int(y_pred), "==", data1_y.iloc[0])
else:
    print("abalone.data(X) =", int(y_pred), "==", data1_y.iloc[0])