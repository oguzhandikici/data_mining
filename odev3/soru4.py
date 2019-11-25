from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
import pandas as pd

train = pd.read_csv('../dataset/mnist_23/mnist_train23.csv')
test = pd.read_csv('../dataset/mnist_23/mnist_test23.csv')

x_train = train.iloc[:, 1:]
y_train = train.iloc[:, 0]

x_test = test.iloc[:, 1:]
y_test = test.iloc[:, 0]

bgrf = BaggingClassifier(RandomForestClassifier(max_features=50), max_samples=50, bootstrap=True)
bgrf.fit(x_train, y_train)
print("Accuracy on Train Data : ", round(bgrf.score(x_train, y_train), 4))
print("Accuracy on Test Data : ", round(bgrf.score(x_test, y_test), 4))
