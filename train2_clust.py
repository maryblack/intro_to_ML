import skimage as sk
from skimage.io import imread
import pylab
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
import statistics as st
import math

#
# class Pixel:
#     def __init__(self, pixel, rgb, cluster):
#         self.pix = pixel
#         self.rgb = rgb
#         self.clust = cluster
#
#      def mean_rgb(self):
#          self.rgb = mean_clust(self.clust)
#
#      def median_rgb(self):
#          self.rgb = median_clust(self.clust)



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

def clust(X, n):
    X = np.array(X)
    kmeans = KMeans(n_clusters=n, init='k-means++', random_state=241).fit(X)
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

def median_rgb(cluster, lists):
    init = lists[cluster]
    R = []
    G = []
    B = []
    for el in init:
        R.append(el[0])
        G.append(el[1])
        B.append(el[2])
    R_mean = st.median(R)
    G_mean = st.median(G)
    B_mean = st.median(B)
    M = np.array([R_mean, G_mean, B_mean])
    return M

def list_median_clust(list_of_cluster, lists):
    median_values = []
    for i in range(0, len(lists)):
        mv = median_rgb(i, lists)
        median_values.append(mv)
    return median_values


def mean_reduction(list_of_cluster, lists,w,h):
    L = np.zeros((w*h,3), dtype='uint')
    mean_values = list_mean_clust(list_of_cluster, lists)
    for i in range (0, len(list_of_cluster)):
        cl = list_of_cluster[i]
        L[i] = mean_values[cl]
    return L

def median_reduction(list_of_cluster, lists, w, h):
    L = np.zeros((w*h,3), dtype='uint')
    median_values = list_median_clust(list_of_cluster, lists)
    for i in range (0, len(list_of_cluster)):
        cl = list_of_cluster[i]
        L[i] = median_values[cl]
    return L

#
# def median_rgb(cluster):

def colour_reduction(image):
    #pylab.imshow(image)
    #pylab.show()
    X, y, w, h = RGB_matrix(image)
    for i in range(12,15):
        list_of_clust = clust(X,i)
        i, lists = by_cluster(list_of_clust, X)
        low_colours = mean_reduction(list_of_clust, lists, w, h)
        #low_colours = median_reduction(list_of_clust, lists, w, h)
        print(f'PSNR:{PSNR(X, low_colours, w,h)} clusters:{i}')

    I = mtx_to_img(low_colours, y, w, h)
    pylab.imshow(I)
    pylab.show()

def MSE(X1, X2, w, h):
    sum_r = 0
    sum_g = 0
    sum_b = 0
    for i in range (0, len(X1)):
        sum_r += (X1[i][0]-X2[i][0])*(X1[i][0]-X2[i][0])
        sum_g += (X1[i][1]-X2[i][1])*(X1[i][1]-X2[i][1])
        sum_b += (X1[i][2]-X2[i][2])*(X1[i][2]-X2[i][2])
    # print(X1[1], X2[1])
    # print(X1[10000], X2[10000])
    # print(sum_r, sum_g, sum_b)
    avg_mse = (sum_r + sum_b + sum_g)/(3*w*h)
    return avg_mse

def PSNR(X1, X2, w, h):
    return 20 * math.log10( 255/ math.sqrt(MSE(X1, X2, w, h)))

def main():
    image = imread('parrots.jpg')
    F = sk.img_as_float(image)
    colour_reduction(image)

    #I = mtx_to_img(X,y,w, h)
    #F = sk.img_as_float(image)
    # pylab.imshow(I)
    # pylab.show()

if __name__ == '__main__':
    main()
