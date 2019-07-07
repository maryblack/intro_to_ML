import pandas as pd
from sklearn.svm import SVC

train = pd.read_csv('svm-data.csv', header=None)


def split_xy(data):
    x_train = data.iloc[:, 1:len(data.columns)]
    y_train = data.iloc[:, 0]
    x_train = x_train.to_numpy()
    y_train = y_train.to_numpy()
    return x_train, y_train


X, y = split_xy(train)
clf = SVC(random_state=241, C=100000, kernel='linear')
clf.fit(X, y)
print(clf.support_)
