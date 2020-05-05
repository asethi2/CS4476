from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import skimage.color as color
from PIL import Image
import numpy as np

def computeQuantizationError(orig_img, quantized_img):
    return np.sum((quantized_img - orig_img) ** 2)

def quantizeRGB(origImage, k):
    random_state = 7

    w, h, d = origImage.shape

    imageArr = np.reshape(origImage, (w * h, d))
    
    km = KMeans(n_clusters=k, random_state=random_state).fit(imageArr)
    outputImgArr = km.predict(imageArr)
    clusterCentres = km.cluster_centers_
    
    outputImg = np.zeros((w, h, d))
    outputImgArrIndex = 0
    for i in range(w):
        for j in range(h):
            outputImg[i][j] = clusterCentres[outputImgArr[outputImgArrIndex]]
            outputImgArrIndex += 1

    return outputImg, clusterCentres


def quantizeHSV(origImage, k):
    random_state = 7

    hsvImage = color.rgb2hsv(origImage / 255)
    w, h, d = hsvImage.shape
    outputImg = np.copy(hsvImage)
    hueArr = np.reshape(np.reshape(hsvImage, (w * h, d))[:, 0], (w * h, 1))

    km = KMeans(n_clusters=k, random_state=random_state).fit(hueArr)
    outputHueArr = km.predict(hueArr)
    clusterCentres = km.cluster_centers_

    outputHueArrIndex = 0
    for i in range(w):
        for j in range(h):
            outputImg[i][j][0] = clusterCentres[outputHueArr[outputHueArrIndex]]
            outputHueArrIndex += 1

    return color.hsv2rgb(outputImg) * 255, clusterCentres



