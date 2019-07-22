import math
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split

def train_test_split_xy(df):
    X = df.iloc[:, 1:len(df.columns)].to_numpy()
    y = df.iloc[:, 0].to_numpy()
    return train_test_split(X, y, test_size=0.8, random_state=241)


def quality_pred(pred):
    return 1 / (1 + math.exp(-pred))


def dev_log(clf, X, y):
    min_log = 100
    min_i = 0
    deviance = []
    for i, y_test in enumerate(clf.staged_decision_function(X)):
        y_pred = [quality_pred(el) for el in y_test]
        test_log = log_loss(y, y_pred)
        if test_log < min_log:
            min_log = test_log
            min_i = i
        deviance.append(test_log)
    return min_log, min_i, deviance


def GB_clf(X_train, X_test, y_train, y_test, lr):
    clf = GradientBoostingClassifier(learning_rate=lr, n_estimators=250, verbose=True, random_state=241)
    clf.fit(X_train, y_train)
    test_min_log, test_min_i, test_deviance = dev_log(clf, X_test, y_test)
    train_min_log, train_min_i, train_deviance = dev_log(clf, X_train, y_train)
    print(f'minimum test: {test_min_log} {test_min_i}')
    print(f'minimum train: {train_min_log} {train_min_i}')
    #plt.plot(test_deviance, linewidth=2, label='learning_rate: {lr}')
    #plt.plot(train_deviance, linewidth=2)

def RF_clf(X_train, X_test, y_train, y_test):
    clf = RandomForestClassifier(n_estimators=36, random_state=241)
    clf.fit(X_train, y_train)
    y_pred = clf.predict_proba(X_test)
    print(f'RFC log__loss: {log_loss(y_test,y_pred)}')


def main():
    df = pd.read_csv('gbm-data.csv')
    print(df.head())
    learning_rate = [1, 0.5, 0.3, 0.2, 0.1]
    X_train, X_test, y_train, y_test = train_test_split_xy(df)
    # print(X_train, X_test, y_train, y_test)
    # plt.figure()
    # for rate in learning_rate:
    #     GB_clf(X_train, X_test, y_train, y_test, rate)
    # plt.show()

    GB_clf(X_train, X_test, y_train, y_test, 0.2)#0.53 36
    RF_clf(X_train, X_test, y_train, y_test)#0.54


if __name__ == '__main__':
    main()
