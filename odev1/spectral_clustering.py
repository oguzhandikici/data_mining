from sklearn.cluster import SpectralClustering
from sklearn.datasets import load_iris

irisData = load_iris()
X = irisData.data
Y = irisData.target

clustering = SpectralClustering(n_clusters=3, assign_labels="discretize", random_state=0).fit(X)
print(clustering.labels_)

