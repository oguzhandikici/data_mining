import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
from sklearn import preprocessing

# load the iris dataset
irisData = load_iris()
print(irisData.data.shape)
# separate the data from the target attributes
X = irisData.data
Y = irisData.target

target_names = irisData.target_names
print(target_names)

# Normalizing Data
normalize_X = preprocessing.normalize(X)
print(normalize_X)

# PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(normalize_X)
print(principalComponents)

# Co-Variance
mean_vec = np.mean(normalize_X, axis=0)
cov_mat = (normalize_X - mean_vec).T.dot((normalize_X - mean_vec)) / (normalize_X.shape[0]-1)
print('\nCovariance Matrix \n-------------------\n{}'.format(cov_mat))

cov_mat = np.cov(normalize_X.T)

eig_vals, eig_vecs = np.linalg.eig(cov_mat)

print('Eigenvectors \n\n-------------------\n{}'.format(eig_vecs))
print('\nEigenvalues \n%\n-------------------\n{}'.format(eig_vals))

pca = PCA(n_components=2)
normalized_data_r = pca.fit(normalize_X).transform(normalize_X)
print(normalized_data_r)


#


print('explained variance ratio (first two components): %s'
      % str(pca.explained_variance_ratio_))

plt.figure()
colors = ['navy', 'turquoise', 'darkorange']
lw = 2

#

for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(normalized_data_r[Y == i, 0], normalized_data_r[Y == i, 1], color=color, alpha=.8, lw=lw, label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('PCA of IRIS dataset')
plt.show()
