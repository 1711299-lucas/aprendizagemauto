from matplotlib import pyplot as plt
import LinearRegression
import pickle as p1
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
column_names = ['comprimento', 'diametro', 'altura', 'peso_inteiro', 'peso_sem_casca', 'peso_viceras', 'peso_concha', 'n_aneis']#input
data = pd.read_csv('abalone.data', header=None, names=column_names)

print(data.head())
data['n_aneis'] = data['n_aneis'].astype(int)
print(data.info())
print(data.describe())
correlation_matrix = data.corr()
print(correlation_matrix)
plt.figure(figsize=(0, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Matriz de Correlação')
plt.show()
X = data.drop(columns=['n_aneis'])
y = data['n_aneis']#output
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f'Tamanho do conjunto de treinamento: {X_train.shape[0]}')
print(f'Tamanho do conjunto de teste: {X_test.shape[0]}')
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'MSE: {mse}')
print(f'R²: {r2}')

train_data=data[:3133]
data_X=train_data.iloc[:,1:8]
data_Y=train_data.iloc[:,8:9]
print(data_X)
print(data_Y)
regr = linear_model.LinearRegression()
preditor_linear_model=regr.fit(data_X, data_Y)
preditor_Pickle = open('../white-wine_quality_predictor', 'wb')
print("white-wine_quality_predictor")
p1.dump(preditor_linear_model, preditor_Pickle)

# Dados de Teste
#import.... + data =
evaluation_data=data[3133:]
data_X=evaluation_data.iloc[:,0:8]
data_Y=evaluation_data.iloc[:,7:8]
print(type(evaluation_data))
print(type(data_X))
loaded_model = p1.load(open('../white-wine_quality_predictor', 'rb'))