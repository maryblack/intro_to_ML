import math
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import ensemble
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
import numpy as np



def train_test_split_xy(df):
    X = df.iloc[:, 1:len(df.columns)].to_numpy()
    y = df.iloc[:, 0].to_numpy()
    return train_test_split(X, y, test_size=0.8, random_state=241)


def quality_pred(pred):
    return 1 / (1 + math.exp(-pred))


def plot_log(test_loss, train_loss):
    plt.figure()
    plt.plot(test_loss, 'r', linewidth=2)
    plt.plot(train_loss, 'g', linewidth=2)
    plt.legend(['test', 'train'])
    plt.show()


def GB_clf(X_train, X_test, y_train, y_test, lr):
    clf = GradientBoostingClassifier(learning_rate=lr, n_estimators=250, verbose=True, random_state=241)
    clf.fit(X_train, y_train)
    y_pred_train = clf.predict_proba(X_train)
    y_pred_test = clf.predict_proba(X_test)
    clf.staged_decision_function(X_test)

    min_log = 100
    min_i = 0
    test_deviance = []
    for i, y_pred_test in enumerate(clf.staged_decision_function(X_test)):
        y_pred = [quality_pred(el) for el in y_pred_test]
        test_log = log_loss(y_test, y_pred)
        if test_log<min_log:
            min_log = test_log
            min_i = i
        test_deviance.append(test_log)

    train_deviance = []
    for i, y_pred_train in enumerate(clf.staged_decision_function(X_train)):
        y_pred = [quality_pred(el) for el in y_pred_train]
        train_log = log_loss(y_train, y_pred)
        train_deviance.append(train_log)
    print(f'minimum: {min_i}  {min_log}')
    #plt.plot(test_deviance, linewidth=2, label='learning_rate: {lr}')
    #plt.plot(train_deviance, linewidth=2, , label=label)






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

    plt.figure()
    GB_clf(X_train, X_test, y_train, y_test, 0.2)#36  0.53
    plt.show()



if __name__ == '__main__':
    main()
