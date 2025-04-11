import matplotlib.pyplot as plt
import pickle as p1
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
data = pd.read_csv("./Dataset/abalone.data", sep=",") ## -- ou -- sep=",")
evaluation_data=data[1001:]
data_X=evaluation_data.iloc[:,1:8]
data_Y=evaluation_data.iloc[:,8:9]
print(type(evaluation_data))
print(type(data_X))
loaded_model = p1.load(open('./abaloneprevisao', 'rb'))
print("Coefficients: \n", loaded_model.coef_)
y_pred=loaded_model.predict(data_X)
z_pred=y_pred-data_Y
