import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("../../dataset/Wine.csv")
binary_data = df.loc[df['Wine Class'] != 3]

# X, y olarak ayrılır.
X = binary_data.iloc[:, 1:]
y = binary_data.iloc[:, 0].values


# Standardizasyon
scaler = StandardScaler().fit(X)
standardized_X = scaler.transform(X)

# Train-Test ayrımı
X_train, X_test, y_train, y_test = train_test_split(standardized_X, y, test_size=0.20)


# Tek Karar Ağacı
def tree(depth, feature):

    decision_tree = DecisionTreeClassifier(max_depth=depth, max_features=feature)
    decision_tree.fit(X_train, y_train)

    train_score = round(decision_tree.score(X_train, y_train), 4)
    test_score = round(decision_tree.score(X_test, y_test), 4)
    auc_score = round(roc_auc_score(y_test, decision_tree.predict(X_test)), 4)

    print('Result for: \nDepth =', depth, ', Feature =', feature)
    print("Accuracy on Train Data : ", train_score)
    print("Accuracy on Test Data : ", test_score)
    print("AUC : ", auc_score)


# Torbalama Karar Ağacı
def bagging_tree(n, bag_sample, bag_feature, tree_depth, tree_feature):

    bag_tree = BaggingClassifier(DecisionTreeClassifier(max_depth=tree_depth, max_features=tree_feature), n_estimators=n, max_samples=bag_sample, max_features=bag_feature)
    bag_tree.fit(X_train, y_train)

    train_score = round(bag_tree.score(X_train, y_train), 4)
    test_score = round(bag_tree.score(X_test, y_test), 4)
    auc_score = round(roc_auc_score(y_test, bag_tree.predict(X_test)), 4)

    print('Result for:\nTrees = {} , Bag Samples = {} , Bag Features = {}, Tree Depth = {}, Tree Features = {}'.format(n, bag_sample, bag_feature, tree_depth, tree_feature))
    print("Accuracy on Train Data : ", train_score)
    print("Accuracy on Test Data : ", test_score)
    print("AUC : ", auc_score)


# Rastgele Orman
def random_forest(n, tree_depth, tree_feature):

    random_forest = RandomForestClassifier(n_estimators=n, max_depth=tree_depth, max_features=tree_feature)
    random_forest.fit(X_train, y_train)

    train_score = round(random_forest.score(X_train, y_train), 4)
    test_score = round(random_forest.score(X_test, y_test), 4)
    auc_score = round(roc_auc_score(y_test, random_forest.predict(X_test)), 4)

    print('Result for:\nTrees = {} , Tree Depth = {}, Tree Features = {}'.format(n, tree_depth, tree_feature))
    print("Accuracy on Train Data : ", train_score)
    print("Accuracy on Test Data : ", test_score)
    print("AUC : ", auc_score)


# Rastgele Orman
def adaboost_tree(n, tree_depth, tree_feature, learn):

    adaboost_tree = AdaBoostClassifier(DecisionTreeClassifier(max_depth=tree_depth, max_features=tree_feature), n_estimators=n, learning_rate=learn)
    adaboost_tree.fit(X_train, y_train)

    train_score = round(adaboost_tree.score(X_train, y_train), 4)
    test_score = round(adaboost_tree.score(X_test, y_test), 4)
    auc_score = round(roc_auc_score(y_test, adaboost_tree.predict(X_test)), 4)

    print('Result for:\nTrees = {} , Tree Depth = {}, Tree Features = {}, Learning Rate'.format(n, tree_depth, tree_feature))
    print("Accuracy on Train Data : ", train_score)
    print("Accuracy on Test Data : ", test_score)
    print("AUC : ", auc_score)