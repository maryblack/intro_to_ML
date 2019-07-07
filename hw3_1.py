from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
from sklearn.preprocessing import scale
from sklearn.model_selection import cross_val_score

def accuracy(model,X,y):
    m = cross_val_score(neigh, X, y, cv=kf.split(X))
    return(m.mean())

def knn(k, X, y):
    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(X, y)
    return neigh

data = pd.read_csv('wine.data', header = None)
#print(data.head())
data = data.to_numpy()
y = [row[0] for row in data]
X = np.delete(data,np.s_[0],1)
#print(y)
#print(X)
X = scale(X)#нормирование

kf = KFold(n_splits=5, random_state=42, shuffle=True)
kf.get_n_splits(X,y)
max_a = 0
max_k = 0
for k in range (1,51):
    neigh = knn(k,X,y)
    a =  accuracy(neigh,X,y)
    if a>max_a:
        max_a = a
        max_k = k
        print(max_k, max_a)

#print(max_k, max_a)


