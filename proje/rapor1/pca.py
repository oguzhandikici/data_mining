import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import preprocessing

df = pd.read_csv("../datasets/Wine.csv")
X = df.iloc[:, 0:4]
y = df.Class

# Normalizing Data
normalized_X = preprocessing.normalize(X)
print(normalized_X)

# PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(normalized_X)
print(principalComponents)

for color, i, target_name in zip(['navy', 'turquoise', 'darkorange'], [0, 1, 2], [1, 2, 3]):
    plt.scatter(principalComponents[y == i, 0], principalComponents[y == i, 1], color=color, alpha=.8, lw=2, label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('PCA of Wine Dataset')
plt.show()
