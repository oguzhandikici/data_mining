from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

iris = datasets.load_iris()
X = iris.data[:, :2]
y = iris.target

plt.scatter(X[:, 0], X[:, 1], c=y, cmap='gist_rainbow')
plt.xlabel('Sepal Length', fontsize=18)
plt.ylabel('Sepal Width', fontsize=18)

km = KMeans(n_clusters=2, n_jobs=4, random_state=42)
km.fit(X)
print(y)

centers = km.cluster_centers_
print(centers)

new_labels = km.labels_
print(new_labels)

# Buradaki sanat tamamen bana ait (MSE)
se = 0

for i in range(len(new_labels)):
    se += (X[i][1] - centers[new_labels[i]][1])**2
mse = se / len(new_labels)
print('Mean Squared Error:  ', mse)


fig, axes = plt.subplots(1, 2, figsize=(16, 8))

axes[0].scatter(X[:, 0], X[:, 1], c=y, cmap='gist_rainbow', edgecolor='k', s=150)
axes[1].scatter(X[:, 0], X[:, 1], c=new_labels, cmap='gist_rainbow', edgecolor='k', s=150)
axes[0].set_xlabel('Sepal length', fontsize=18)
axes[0].set_ylabel('Sepal width', fontsize=18)
axes[1].set_xlabel('Sepal length', fontsize=18)
axes[1].set_ylabel('Sepal width', fontsize=18)
axes[0].tick_params(direction='in', length=10, width=5, colors='k', labelsize=20)
axes[1].tick_params(direction='in', length=10, width=5, colors='k', labelsize=20)
axes[0].set_title('Actual', fontsize=18)
axes[1].set_title('Predicted', fontsize=18)
plt.show()
