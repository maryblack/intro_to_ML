import pandas as pd
from sklearn.decomposition import PCA
from numpy import corrcoef



def pca_comp(X,components):
    pca = PCA(n_components=components)
    pca.fit(X)
    return pca.explained_variance_ratio_

def first_comp(X):
    pca = PCA(n_components=10)
    pca.fit(X)
    pca.transform(X)
    return pca.transform(X)[:, 0]

def dispersion(X, treshold):
    for i in range(1,30):
        s = sum(pca_comp(X,i))
        if s >= 0.9:
            print(f'{i}: {s}')
            break
def best_company(X):
    l = len(X.columns)
    pca = PCA(n_components=10)
    pca.fit(X)
    names = X.columns.values
    comp = pca.components_[0]
    max_n = ''
    max_v = comp[0]
    for i in range (0,l):
        if comp[i]>max_v:
            max_v = comp[i]
            max_n = names[i]

    return max_n, max_v


def main():
    prices = pd.read_csv('close_prices.csv')
    djia = pd.read_csv('djia_index.csv')
    DJA = djia['^DJI']
    X = prices.iloc[:, 1:len(prices.columns)]
    dispersion(X,0.9)#4
    print(corrcoef(first_comp(X),DJA))#0.91
    print(best_company(X))#V
    print(X.columns.values)


if __name__ == '__main__':
    main()
