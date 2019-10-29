import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import preprocessing


df = pd.read_csv("../datasets/Wine.csv")

X = df.iloc[:, 0:4]
y = df.Class
normalized_X = preprocessing.normalize(X)

plt.scatter(X[:, 0], X[:, 1], c=y, cmap='gist_rainbow')

km = KMeans(n_clusters=2, n_jobs=4, random_state=42)
km.fit(normalized_X)

centers = km.cluster_centers_

new_labels = km.labels_

# Buradaki sanat tamamen bana ait (MSE)
se = 0

for i in range(len(new_labels)):
    se += (X[i][1] - centers[new_labels[i]][1])**2
mse = se / len(new_labels)
print('Mean Squared Error:  ', mse)


fig, axes = plt.subplots(1, 2, figsize=(16, 8))

axes[0].scatter(X[:, 0], X[:, 1], c=y, cmap='gist_rainbow', edgecolor='k', s=150)
axes[1].scatter(X[:, 0], X[:, 1], c=new_labels, cmap='gist_rainbow', edgecolor='k', s=150)
axes[0].tick_params(direction='in', length=10, width=5, colors='k', labelsize=20)
axes[1].tick_params(direction='in', length=10, width=5, colors='k', labelsize=20)
axes[0].set_title('Actual', fontsize=18)
axes[1].set_title('Predicted', fontsize=18)
plt.show()
