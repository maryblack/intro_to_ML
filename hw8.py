import pandas as pd
import numpy as np
from sklearn import metrics as m
from  sklearn.metrics import roc_auc_score


def error_table(df):
    true = df['true']
    pred = df['pred']
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    for i in range (0,len(true)):
        if true[i]==1:
            if true[i]==pred[i]:
                TP += 1
            else:
                FN += 1
        elif true[i]==0:
            if true[i]==pred[i]:
                TN += 1
            else:
                FP += 1
    return TP, FP, FN, TN

def PRF_scores(TP, FP, FN, TN):
    P = TP/(TP+FP)
    R = TP/(TP+FN)
    F = 2*P*R/(P+R)
    return P, R, F

def all_scores(true, pred):
    acc = m.accuracy_score(true, pred)
    P = m.precision_score(true,pred)
    R = m.recall_score(true,pred)
    F = m.f1_score(true,pred)
    return round(acc,2), round(P,2), round(R,2), round(F,2)

def PR_treshold(true, pred, tr):
    curve = m.precision_recall_curve(true,pred)
    precision = curve[0]
    recall = curve[1]
    max_p = precision[0]
    for i in range(0, len(recall)):
        if recall[i]>=tr:
            if precision[i]>max_p:
                max_p = precision[i]
    return max_p

def scores_clf(data, tr):
    true = data['true']
    logreg = data['score_logreg']
    svm = data['score_svm']
    knn = data['score_knn']
    tree = data['score_tree']
    print( f'___Logreg___\nAUC, Precision: {roc_auc_score(true,logreg)} {PR_treshold(true,logreg,tr)} ')
    print( f'___SVM___\nAUC, Precision: {roc_auc_score(true,svm)} {PR_treshold(true,svm,tr)} ')
    print( f'___KNN___\nAUC, Precision: {roc_auc_score(true,knn)} {PR_treshold(true,knn,tr)} ')
    print( f'___TREE___\nAUC, Precision: {roc_auc_score(true,tree)} {PR_treshold(true,tree,tr)} ')


def main():
    df = pd.read_csv('classification.csv')
    scores = pd.read_csv('scores.csv')
    true = df['true']
    pred = df['pred']
    print(scores.head())
    TP, FP, FN, TN = error_table(df)
    print(f'TP, FP, FN, TN: {TP} {FP} {FN} {TN}')
    acc, P, R, F = all_scores(true, pred)
    print(f'accuracy, precision, recall, f-score: {acc} {P} {R} {F}')
    scores_clf(scores, 0.7)

if __name__ == '__main__':
    main()
