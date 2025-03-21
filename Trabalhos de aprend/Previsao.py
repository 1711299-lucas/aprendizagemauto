import matplotlib.pyplot as plt
import pickle as p1
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

data_x=input("introduza valores do abalone: \n")
data=data_x.split(";")
print(data)
fmap_data = map(float, data)
print(fmap_data)

flist_data =    np.array([list(fmap_data)])

print(flist_data)
data1 = pd.read_csv("abalone.data",sep=";")
data2 = data1.iloc[:, :11]
data_preparation = pd.DataFrame(flist_data, columns=data1.columns[:11])
out=data2
for x in out:
    print(x,data_preparation[x].values)
loaded_model = p1.load(open('abalone.data', 'rb'))
y_pred=loaded_model.predict(flist_data)
print("wine quality",int(y_pred))

if int(y_pred)  ==  data1_y[0]:
    print("abalone.data(X)=", int(y_pred), "==", data1_y[0])
else:
    print("abalone.data(X)=", int(y_pred), "==", data1_y[0])

