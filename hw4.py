import pandas as pd
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

train = pd.read_csv('train.csv', header=None)
test = pd.read_csv('test.csv', header=None)


def split_xy(data):
    x_train = data.iloc[:, 1:len(data.columns)]
    y_train = data.iloc[:, 0]
    x_train = x_train.to_numpy()
    y_train = y_train.to_numpy()
    return x_train, y_train


X_train, y_train = split_xy(train)
X_test, y_test = split_xy(test)

clf = Perceptron()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
a1 = accuracy_score(y_test, y_pred)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
clf = Perceptron()
clf.fit(X_train_scaled, y_train)
y_pred = clf.predict(X_test_scaled)
a2 = accuracy_score(y_test, y_pred)

print(a1, a2, a2 - a1)
