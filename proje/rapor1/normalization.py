import pandas as pd
from sklearn import preprocessing
df = pd.read_csv("../datasets/Wine.csv")

X = df.iloc[:, 1:4]
y = df.Class

normalized_X = preprocessing.normalize(X)
print(normalized_X)

scaler = preprocessing.StandardScaler().fit(X)
standardized_X = scaler.transform(X)
print(standardized_X)
