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
    prod = pca.transform(X)
    return pca.transform(X)[:, 0]

def dispersion(X, treshold):
    for i in range(1,30):
        s = sum(pca_comp(X,i))
        if s >= 0.9:
            print(f'{i}: {s}')
            break

def main():
    prices = pd.read_csv('close_prices.csv')
    djia = pd.read_csv('djia_index.csv')
    DJA = djia['^DJI']
    X = prices.iloc[:, 1:len(prices.columns)]
    dispersion(X,0.9)
    print(corrcoef(first_comp(X),DJA))


if __name__ == '__main__':
    main()
