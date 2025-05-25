import matplotlib.pyplot as plt
import pickle as p1
import numpy as np
import pandas as pd
from fontTools.misc.symfont import y
from sklearn import datasets, linear_model

# Plot a predict point
sns.scatterplot(
x=(X[1,0]+ X[43,0])/2,
y=(X[1,1]+ X[43,1])/2,
marker="X",
s=90,
hue=optdigits.names[y],
palette=cmap_bold,
alpha=1.0,
edgecolor="w",)
plt.show()
