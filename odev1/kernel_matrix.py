from sklearn.datasets import load_iris
from sklearn import preprocessing
from sklearn.metrics import pairwise as pw


irisData = load_iris()
X = irisData.data
Y = irisData.target
normalize_X = preprocessing.normalize(X)


print(pw.rbf_kernel(normalize_X))
