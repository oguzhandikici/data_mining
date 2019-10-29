from sklearn import svm
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Data okunur.

all_data = pd.read_csv(
    "../dataset/yeast/yeast.data",
    delim_whitespace=True,
    header=None
)

# Okunan dataset X, y olarak ayrılır.

X = all_data[[col for col in range(1, 9)]].values
y = all_data[9].values

# Train ve Test datası ayrılır.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1000)

# Logistic regression ve SVM parametreleri ayarlanıp train datası üstüne fitlenir.

logreg = LogisticRegression(C=1e5, solver='liblinear', multi_class='auto')
SVM = svm.SVC(gamma='scale')
logreg.fit(X_train, y_train)
SVM.fit(X_train, y_train)

# Accuracies


def svm_accuracy(X, y):
    accurate = 0
    for i in range(len(X)):
        if SVM.predict([X[i]]) == y[i]:
            accurate += 1
    return accurate / len(X)


def logreg_accuracy(X, y):
    accurate = 0
    for i in range(len(X)):
        if logreg.predict([X[i]]) == y[i]:
            accurate += 1
    return accurate / len(X)


# Test datası üstünden doğruluk hesaplanır.

print('SVM Accuracy = ', round(svm_accuracy(X_test, y_test), 3))
print('Logistic Regression Accuracy = ', round(logreg_accuracy(X_test, y_test), 3))
