from sklearn import svm
import pandas as pd
from sklearn.metrics import confusion_matrix as cm
from sklearn.metrics import roc_auc_score

# Data okunur.

training_data = pd.read_csv(
    "../dataset/optdigits/optdigits.tra",
    header=None
)

test_data = pd.read_csv(
    "../dataset/optdigits/optdigits.tes",
    header=None
)

# Okunan dataset X, y olarak ayrılır.

X_train = training_data[[col for col in range(64)]].values
y_train = training_data[64].values
X_test = test_data[[col for col in range(64)]].values
y_test = test_data[64].values

# i=classno pozitif, diğerleri negatif olacak şekilde düzenleme yapılır.


def pos_neg(classno):
    global y_test, y_train
    for i in range(len(y_train)):
        if y_train[i] == classno:
            y_train[i] = 1
        else:
            y_train[i] = 0

    for i in range(len(y_test)):
        if y_test[i] == classno:
            y_test[i] = 1
        else:
            y_test[i] = 0


# Accuracy'nin en düşük olduğu i

pos_neg(8)

# SVM parametreleri ayarlanıp train data üstüne fitlenir.

SVM = svm.SVC(kernel="rbf", C=1.0, gamma="scale")
SVM.fit(X_train, y_train)

# Test yapılıp doğruluk oranı bulunur.

print(
    'SVM Accuracy = {}'.format(
        round(SVM.score(X_test, y_test), 3)
    )
)
print(
    'Confusion Matrix =\n {}'.format(
        cm(y_test, SVM.predict(X_test))
    )
)
print(
    'AUC Score = {}'.format(
        round(roc_auc_score(y_test, SVM.predict(X_test)), 3)
    )
)
