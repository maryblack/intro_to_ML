from skimage.io import imread
import skimage as sk
import pylab
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans

def RGB_matrix(image):
    y = []
    h = len(image)
    w = len(image[0])
    X = np.zeros((w*h,3), dtype='uint')

    for j in range(0, h):
        for i in range(0,w):
            y.append([j,i])
            el = image[j][i]
            X[i+j*w] = el
    return X,y,w,h

def mtx_to_img(X,y,w,h):
    rgbArray = np.zeros((h, w, 3), dtype='uint8')
    for j in range(0, h):
        for i in range(0, w):
            ind = i+j*w
            rgbArray[j][i][0] = X[ind][0]
            rgbArray[j][i][1] = X[ind][1]
            rgbArray[j][i][2] = X[ind][2]

    return Image.fromarray(rgbArray, mode='RGB')

def clust(X):
    X = np.array(X)
    kmeans = KMeans(init='k-means++', random_state=241).fit(X)
    return kmeans.labels_

def by_cluster(list_of_clust, X):
    num = max(list_of_clust)
    lists = [[] for i in range(1,num)]
    i = 0
    for i in range(0, len(X)):
        ind = list_of_clust[i]
        lists[ind].append(X[i])
    return lists





def main():
    image = imread('parrots.jpg')

    X,y,w,h = RGB_matrix(image)
    I = mtx_to_img(X,y,w, h)
    F = sk.img_as_float(image)
    print(max(clust(X)))
    clusters = by_cluster(clust(X),X)
    for c in clusters:
        print(c)
    pylab.imshow(I)
    pylab.show()

if __name__ == '__main__':
    main()