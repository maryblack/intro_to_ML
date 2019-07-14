import inline as inline
import math
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
% matplotlib
inline


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


def GB_clf(X_train, X_test, y_train, y_test, lr):
    clf = GradientBoostingClassifier(learning_rate=lr, n_estimators=250, verbose=True, random_state=241)
    clf.fit(X_train, y_train)
    y_pred_train = clf.predict_proba(X_train)
    y_pred_test = clf.predict_proba(X_test)
    # pred = quality_pred(clf.staged_decision_function(X_test))
    test_loss = log_loss(y_test, y_pred_test)
    train_loss = log_loss(y_train, y_pred_train)
    plot_log(test_loss, train_loss)
    # print(f'pred {pred}')


def main():
    df = pd.read_csv('gbm-data.csv')
    print(df.head())
    learning_rate = [1, 0.5, 0.3, 0.2, 0.1]
    X_train, X_test, y_train, y_test = train_test_split_xy(df)
    # print(X_train, X_test, y_train, y_test)
    GB_clf(X_train, X_test, y_train, y_test, learning_rate[0])


if __name__ == '__main__':
    main()
