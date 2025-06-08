import numpy as np
from sklearn.datasets import load_digits
import pickle as p1
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap

kmeans = p1.load(open('digits_predictor_Kmeans1.pkl', 'rb'))
x_test = p1.load(open('X_test.pkl', 'rb'))
y_test = p1.load(open('y_test.pkl', 'rb'))

(n_samples, n_features), n_digits = x_test.shape, np.unique(y_test).size

pca = PCA(n_components=2)
reduced_data = pca.fit_transform(x_test)

y_pred = Kmeans.predict(reduced_data)
print(y_pred)

cmaps = ListedColormap(['r', 'g', 'b'])
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=y_pred, cmap='jet')


centroids = kmeans.cluster_centers_
plt.scatter(
    centroids[:, 0],
    centroids[:, 1],
    marker="o",
    s=90,
    linewidths=1,
    color="b",
    edgecolors="black"
)

plt.show()
