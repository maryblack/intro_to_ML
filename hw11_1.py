import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer



def split_n_prep(df):
    df['Sex'] = df['Sex'].map(lambda x: 1 if x == 'M' else (-1 if x == 'F' else 0))
    print(df.head())
    X = df.iloc[:, 0:(len(df.columns)-2)]
    y = df.iloc[:, len(df.columns)-1]
    X = X.to_numpy()
    y = y.to_numpy()
    return X, y

def cvKfold(X,y):
    kf = KFold(n_splits=5, random_state=1, shuffle=True)
    kf.get_n_splits(X, y)
    return kf

def RF_estimators(df, max_s):
    X, y = split_n_prep(df)
    kf = cvKfold(X,y)

    for e in range(20,51):
        rf = RandomForestRegressor(n_estimators=e, random_state=1)
        cvs = cross_val_score(rf, X, y, cv=kf.split(X), scoring=make_scorer(r2_score))
        r2 = round(cvs.mean(),2)
        #r2 = cvs.mean()
        #print(f'r2: {r2} n_estimators: {e}')
        if r2 > max_s:
            print(f'r2: {r2} n_estimators: {e}')
            break




def main():
     df = pd.read_csv('abalone.csv')
     RF_estimators(df, 0.52)
    # import sklearn
    # print(sklearn.__version__)

if __name__ == '__main__':
    main()

