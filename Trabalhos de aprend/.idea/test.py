import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Carregar os dados
XX = pd.read_csv("../casas IA/casas1.csv")

# Supondo que a primeira coluna seja a área e a segunda coluna seja o preço
X = XX.iloc[:, 0].values.reshape(-1, 1)  # Área
y = XX.iloc[:, 1].values  # Preço

# Criar o modelo de regressão linear
regr = LinearRegression()
regr.fit(X, y)

# Fazer previsões
yz = regr.predict(X).round(3)

# Plotar os dados
plt.title("Preços das casas vs Área em metros quadrados")
plt.xlabel('Área em ($m^2$)')
plt.ylabel('Preço Estimado (¿)')
plt.scatter(X, y, color="black", label="Dados reais")
plt.plot(X, yz, color="blue", linewidth=3, label="Modelo de Regressão")
plt.legend()
plt.show()

# Exibir resultados
print("Preços reais:", y)
print("Preços estimados:", yz)
print("Preço estimado para 300 m²:", regr.predict([[300]]).round(3))
