import  numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
import pandas as pd
import pickle as p1
n_neighbors = 15
# import some data to play with
"""iris = datasets.load_iris()"""
data = pd.read_csv("./Data-set/optdigits.tra", sep=",", header=None)
X = data.iloc[ :, :64]
y = data.iloc[ :, 64:]
y=np.array(y).T[0]

print(X)
print(y)

clf = neighbors.KNeighborsClassifier(n_neighbors, weights="uniform")
clf.fit(X, y)


preditor_Pickle = open('optdigitspredict', 'wb')
print("optdigitspredict")
p1.dump(clf, preditor_Pickle)