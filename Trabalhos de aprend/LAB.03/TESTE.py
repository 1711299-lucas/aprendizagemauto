from functools import reduce
import numpy as np
from sklearn.decomposition import PCA  # Importação do PCA adicionada
from sklearn.datasets import load_digits
import pickle as p1
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split


with open('X_test.pkl', 'rb') as X_test_file:
    X_test = p1.load(X_test_file)
with open('y_test.pkl', 'rb') as y_test_file:
    y_test = p1.load(y_test_file)

reduced_data = PCA(n_components=2).fit_transform(X_test)

kmeanss = KMeans(init="k-means++", n_clusters=np.unique(y_test).size, n_init=4)
kmeans2 = kmeanss.fit(reduced_data)

preditor_Pickle2 = open('digits_predictor_kmeans2.pkl', 'wb')
print("Criação do Preditor:", "digits_predictor_kmeans2.pkl")
p1.dump(kmeans2, preditor_Pickle2)
preditor_Pickle2.close()

h = 0.02

x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z = kmeans2.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

cmaps = ['Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds', 'YlOrBr', 'OrRd', 'PuRd', 'BuPu', 'Gnbu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']
plt.figure("Nome")
plt.clf()
plt.title(
    "K-means clustering on the digits dataset (PCA-reduced data)\n"
    "Centroids are marked with white cross"
)

plt.imshow(
    Z,
    interpolation="nearest",
    extent=(xx.min(), xx.max(), yy.min(), yy.max()),
    cmap=plt.cm.Paired,# ,'jet'
    aspect="auto",
    origin="lower",
)

plt.plot(reduced_data[:, 0], reduced_data[:, 1], "k.", markersize=3)

centroids = kmeans2.cluster_centers_
plt.scatter(
    centroids[:, 0],
    centroids[:, 1],
    marker="x",
    s=100,
    linewidths=3,
    color="w",
    zorder=10,
)

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()
