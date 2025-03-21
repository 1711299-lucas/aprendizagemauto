from matplotlib import pyplot as plt
import pickle as p1
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns

# Definindo os nomes das colunas
column_names = ['comprimento', 'diametro', 'altura', 'peso_inteiro', 'peso_sem_casca', 'peso_viceras', 'peso_concha', 'n_aneis']

# Carregando o conjunto de dados
data = pd.read_csv('abalone.data', header=None, names=column_names)

# Exibindo as primeiras linhas do conjunto de dados
print(data.head())

# Convertendo 'n_aneis' para o tipo inteiro
data['n_aneis'] = data['n_aneis'].astype(int)

# Exibindo informações e estatísticas do conjunto de dados
print(data.info())
print(data.describe())

# Calculando e exibindo a matriz de correlação
correlation_matrix = data.corr()
print(correlation_matrix)

# Plotando o heatmap da matriz de correlação
plt.figure(figsize=(8, 6))  # Corrigido para um tamanho de figura válido
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Matriz de Correlação')
plt.show()

# Preparando os dados para treinamento e teste
X = data.drop(columns=['n_aneis'])
y = data['n_aneis']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Exibindo os tamanhos dos conjuntos de treinamento e teste
print(f'Tamanho do conjunto de treinamento: {X_train.shape[0]}')
print(f'Tamanho do conjunto de teste: {X_test.shape[0]}')

# Treinando o modelo de regressão linear
model = linear_model.LinearRegression()
model.fit(X_train, y_train)

# Fazendo previsões
y_pred = model.predict(X_test)

# Calculando e exibindo métricas de desempenho
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'MSE: {mse}')
print(f'R²: {r2}')

# Salvando o modelo treinado usando pickle
with open('abalone_quality_predictor.pkl', 'wb') as preditor_Pickle:
    p1.dump(model, preditor_Pickle)

# Dados de Teste
evaluation_data = data.iloc[3133:]  # Corrigido para usar iloc
data_X = evaluation_data.drop(columns=['n_aneis'])  # Corrigido para usar drop
data_Y = evaluation_data['n_aneis']  # Corrigido para selecionar a coluna correta

# Carregando o modelo para avaliação
with open('abalone_quality_predictor.pkl', 'rb') as preditor_Pickle:
    loaded_model = p1.load(preditor_Pickle)

# Exibindo os coeficientes do modelo carregado
print("Coefficients: \n", loaded_model.coef_)

# Fazendo previsões nos dados de avaliação
y_pred_eval = loaded_model.predict(data_X)

# Calculando a diferença entre previsões e valores reais
z_pred = y_pred_eval - data_Y.values  # Corrigido para garantir que data_Y seja um array numpy

# Exibindo as previsões e as diferenças
print("Predictions: \n", y_pred_eval)
print("Differences: \n", z_pred)