y_train_6 = []
X_train_6 = []

for i in range(len(y_train)):
    if y_train[i] == 6:
        y_train_6.append(list(y_train)[i])
        X_train_6.append(list(X_train)[i])

y_train_6 = np.array(y_train_6)
X_train_6 = np.array(X_train_6)
print(X_train_6)
print(y_train_6)