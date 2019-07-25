import skimage as sk
from skimage.io import imread
import pylab
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
import statistics as st


class Pixel:
    def __init__(self, pixel, rgb, cluster):
        self.pix = pixel
        self.rgb = rgb
        self.clust = cluster

    # def mean_rgb(self):
    #     self.rgb = mean_clust(self.clust)
    #
    # def median_rgb(self):
    #     self.rgb = median_clust(self.clust)



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
    num = max(list_of_clust)+1
    lists = [[] for i in range(0, num)]
    i = 0
    for i in range(0, len(X)):
        ind = list_of_clust[i]
        lists[ind].append(X[i])
    return num, lists

def mean_rgb(cluster, lists):
    init = lists[cluster]
    R = []
    G = []
    B = []
    for el in init:
        R.append(el[0])
        G.append(el[1])
        B.append(el[2])
    R_mean = st.mean(R)
    G_mean = st.mean(G)
    B_mean = st.mean(B)
    M = np.array([R_mean, G_mean, B_mean])
    return M

def list_mean_clust(list_of_cluster, lists):
    mean_values = []
    for i in range(0, len(lists)):
        mv = mean_rgb(i, lists)
        mean_values.append(mv)
    return mean_values

def reduction(X,list_of_cluster, mean_values):
    for i in range (0, len(list_of_cluster)):
        cl = list_of_cluster[i]
        X[i] = mean_values[cl]

    return X

#
# def median_rgb(cluster):

def colour_reduction(image):
    pylab.imshow(image)
    pylab.show()
    X,y,w,h = RGB_matrix(image)
    list_of_clust = clust(X)
    num, lists = by_cluster(list_of_clust, X)
    mean_values = list_mean_clust(list_of_clust, lists)
    low_colours = reduction(X, list_of_clust, mean_values)
    I = mtx_to_img(low_colours, y, w, h)
    pylab.imshow(I)
    pylab.show()


def main():
    image = imread('parrots.jpg')
    #F = sk.img_as_float(image)
    colour_reduction(image)

    #I = mtx_to_img(X,y,w, h)
    #F = sk.img_as_float(image)
    # pylab.imshow(I)
    # pylab.show()

if __name__ == '__main__':
    main()
