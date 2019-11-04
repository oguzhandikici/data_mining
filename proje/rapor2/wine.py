import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score


# Data okunur (Data içinden sadece 1,2 classları seçilir).
raw_data = pd.read_csv("../dataset/Wine.csv")
binary_data = raw_data.loc[raw_data['Wine Class'] != 3]


# X, y olarak ayrılır.
X = binary_data.drop(["Wine Class"], axis=1)
y = binary_data["Wine Class"].values

# Normalizasyon
scaler = MinMaxScaler(feature_range=(0, 1))
X = scaler.fit_transform(X)
# 10 kat çapraz geçerleme (her yöntemde kullanılacak).
kf = StratifiedKFold(n_splits=10)


# Logistic Regression
logreg = LogisticRegression(C=1e3, solver='liblinear', multi_class='auto')
logreg_scores = []
logreg_auc_scores = []
for train_index, test_index in kf.split(X, y):
    X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
    logreg.fit(X_train, y_train)
    logreg_scores.append(logreg.score(X_test, y_test))
    logreg_auc_scores.append(roc_auc_score(y_test, logreg.predict(X_test)))
print(
    'Logistic Regression Score = {}, AUC Score = {}'.format(
        round(sum(logreg_scores)/len(logreg_scores), 3),
        round(sum(logreg_auc_scores)/len(logreg_auc_scores), 3)
    )
)

# KNN (Her train ve test data için en iyi olan k değeri bulunur)
knn_k_scores = []
knn_scores = []
knn_k_auc_scores = []
knn_auc_scores = []

for train_index, test_index in kf.split(X, y):
    X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
    for i in range(1, 20):
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(X_train, y_train)
        knn_k_scores.append(knn.score(X_test, y_test))
        knn_auc_scores.append(roc_auc_score(y_test, knn.predict(X_test)))
    knn_scores.append(max(knn_k_scores))
    knn_auc_scores.append(max(knn_auc_scores))
    knn_k_scores = []
    knn_k_auc_scores = []

print(
    'Max KNN Score = {}, Max AUC Score = {}'.format(
        round(sum(knn_scores)/len(knn_scores), 3),
        round(sum(knn_auc_scores)/len(knn_auc_scores), 3)
    )
)

# SVM
svm_scores = []
svm_auc_scores = []
SVM = SVC(C=100, kernel="linear", random_state=1)

for train_index, test_index in kf.split(X, y):
    X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
    SVM.fit(X_train, y_train)
    svm_scores.append(SVM.score(X_test, y_test))
    svm_auc_scores.append(roc_auc_score(y_test, SVM.predict(X_test)))

print(
    "SVM Score = {}, AUC Score = {}".format(
        round(sum(svm_scores)/len(svm_scores), 3),
        round(sum(svm_auc_scores)/len(svm_auc_scores), 3)
    )
)

# Naive Bayes
nb_scores = []
nb_auc_scores = []
nb = GaussianNB()
for train_index, test_index in kf.split(X, y):
    X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
    nb.fit(X_train, y_train)
    nb_scores.append(nb.score(X_test, y_test))
    nb_auc_scores.append(roc_auc_score(y_test, nb.predict(X_test)))
print(
    "Naive Bayes Score = {}, AUC Score = {}".format(
        round(sum(nb_scores)/len(nb_scores), 3),
        round(sum(nb_auc_scores)/len(nb_auc_scores), 3)
    )
)
