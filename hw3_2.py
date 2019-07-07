import numpy as np
from sklearn import datasets
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import scale


def accuracy(model, X, y, c_v):
    y_predict = model.predict(X)
    # scor = mean_squared_error(y, y_predict)
    m = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
    # print(m)
    return m.mean()


ds = datasets.load_boston()
y = ds['target']

X = scale(ds.data)
kf = KFold(n_splits=5, random_state=42, shuffle=True)
kf.get_n_splits(X, y)

c_v = kf.split(X)

p_array = np.linspace(1.0, 10.0, num=200)
max_p = p_array[0]
max_a = -100
for p_v in p_array:
    neigh = KNeighborsRegressor(n_neighbors=5, weights='distance', metric='minkowski', p=p_v)
    neigh.fit(X, y)
    a = accuracy(neigh, X, y, c_v)
    if a > max_a:
        max_a = a
        max_p = p_v

print(max_p, max_a)
