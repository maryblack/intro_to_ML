import pandas as pd
import numpy as np
import math

def scalar_prod(w1,w2):
    s = 0
    for i in range(0,len(w1)):
        s+= w1[i]*w2[i]
    return s


def r_exp(x_i1, x_i2, y_i,w1,w2):
    return 1 - 1/(1+math.exp(-y_i*(w1*x_i1+w2*x_i2)))

def a_sigma(x, w):
    return 1/(1+math.exp(-w[0]*x[0]-w[1]*x[1]))


def sum1_l(x, y, w1, w2):
    s=0
    x1 = x[:,0]
    x2 = x[:,1]
    l = len(y)
    for i in range(0,l):
        s+=y[i]*x1[i]*r_exp(x1[i],x2[i],y[i],w1,w2)
    return s/l

def sum2_l(x, y, w1, w2):
    s=0
    x1 = x[:,0]
    x2 = x[:,1]
    l = len(y)
    for i in range(0,l):
        s+=y[i]*x2[i]*r_exp(x1[i],x2[i],y[i],w1,w2)
    return s/l


def normal_grad(X,y, w_0:list, k):
    w1 = w_0[0]
    w2 = w_0[1]
    w1 = w1 + k*sum1_l(X,y,w1,w2)
    w2 = w2 + k*sum2_l(X,y,w1,w2)
    w = [w1,w2]
    wx = [w[i]-w_0[i] for i in range(0,len(w))]
    eps = math.sqrt(scalar_prod(wx,wx))
    return w, eps

def LogRegression(X,y,k):
    w_0 = [0,0]#initial weights
    e = 1e-5
    N = 1e+4#iterations
    i = 0
    eps = 10
    while (eps > e) and (i < N):
        w, eps = normal_grad(X,y,w_0,k)
        w_0 = w
        i+=1

    #a_x = [a_sigma(x,w) for x in X]
    a_x = [np.sign(scalar_prod(x,w)) for x in X]
    return a_x


def L2_grad(X,y):
    y_pred = 0
    return y_pred

def split_xy(data):
    x = data.iloc[:, 1:len(data.columns)]
    y = data.iloc[:, 0]
    x = x.to_numpy()
    y = y.to_numpy()
    return x, y

def main():
    df = pd.read_csv('data-logistic.csv', header=None)
    X, y = split_xy(df)
    print(X)
    print(y)
    print(LogRegression(X,y,0.1))


if __name__ == '__main__':
    main()
