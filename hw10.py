import pandas as pd
from sklearn.decomposition import PCA

def pca_comp(X,components):
    pca = PCA(n_components=components)
    pca.fit(X)
    return pca.explained_variance_ratio_

def main():
    prices = pd.read_csv('close_prices.csv')
    X = prices.iloc[:, 1:len(prices.columns)]
    for i in range(1,30):
        s = sum(pca_comp(X,i))
        if s >= 0.9:
            print(f'{i}: {s}')
            break

if __name__ == '__main__':
    main()
