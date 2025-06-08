import numpy as np
from sklearn.datasets import load_digits
import pickle as p1
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

X, y = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.33, random_state=42)
(n_samples, n_features), n_digits = X_train.shape, np.unique(y_train).size
print(f"{n_digits}; {n_samples}; {n_features}")
kmeans = KMeans(init="k-means++", n_clusters=n_digits, n_init=4, random_state=0)
Kmeans = kmeans.fit(X_train)
print(Kmeans, Kmeans.score(X_test))
predictor_Pickle = open('digits_predictor_Kmeans1.pkl', 'wb')
print("Criação do Predictor:", "digits_predictor_Kmeans1.pkl")
p1.dump(Kmeans, predictor_Pickle)
predictor_Pickle.close()
X_test1 = open('X_test.pkl', 'wb')
p1.dump(X_test, X_test1)
X_test1.close()
y_test1 = open('y_test.pkl', 'wb')
p1.dump(y_test, y_test1)
y_test1.close()