import numpy as np
import cv2
from proj3_code.student_harris import get_gradients

def get_magnitudes_and_orientations(dx, dy):
    """
    This function will return the magnitudes and orientations of the
    gradients at each pixel location. 
    Args:
    -   dx: A numpy array of shape (m,n), representing x gradients in the image
    -   dy: A numpy array of shape (m,n), representing y gradients in the image

    Returns:
    -   magnitudes: A numpy array of shape (m,n), representing magnitudes of the
            gradients at each pixel location
    -   orientations: A numpy array of shape (m,n), representing angles of
            the gradients at each pixel location. angles should range from 
            -PI to PI.
 
    """
    magnitudes = (dx**2 + dy**2)**(1/2)
    # print(magnitudes)
    orientations = np.arctan2(dy, dx)
    # print(orientations)
    return magnitudes, orientations

def get_feat_vec(x,y,magnitudes, orientations,feature_width):
    """
    This function returns the feature vector for a specific interest point.
    To start with, you might want to simply use normalized patches as your
    local feature. This is very simple to code and works OK. However, to get
    full credit you will need to implement the more effective SIFT descriptor
    (See Szeliski 4.1.2 or the original publications at
    http://www.cs.ubc.ca/~lowe/keypoints/)

    Your implementation does not need to exactly match the SIFT reference.
    Here are the key properties your (baseline) descriptor should have:
    (1) a 4x4 grid of cells, each feature_width/4. It is simply the
        terminology used in the feature literature to describe the spatial
        bins where gradient distributions will be described. The grid will extend
        feature_width/2 to the left of the "center", and feature_width/2 - 1 to the right
    (2) each cell should have a histogram of the local distribution of
        gradients in 8 orientations. Appending these histograms together will
        give you 4x4 x 8 = 128 dimensions. The bin centers for the histogram 
        should be at -7pi/8,-5pi/8,...5pi/8,7pi/8. The histograms should be added
        to the feature vector left to right then row by row (reading order).  
    (3) Each feature should be normalized to unit length.
    (4) Each feature should be raised to a power less than one(use .9)

    For our tests, you do not need to perform the interpolation in which each gradient
    measurement contributes to multiple orientation bins in multiple cells
    As described in Szeliski, a single gradient measurement creates a
    weighted contribution to the 4 nearest cells and the 2 nearest
    orientation bins within each cell, for 8 total contributions. This type
    of interpolation probably will help, though, so feel free to try it.
    The autograder will only check for each gradient contributing to a single bin.
    

    Args:
    -   x: a float, the x-coordinate of the interest point
    -   y: A float, the y-coordinate of the interest point
    -   magnitudes: A numpy array of shape (m,n), representing image gradients
            at each pixel location
    -   orientations: A numpy array of shape (m,n), representing gradient
            orientations at each pixel location
    -   feature_width: integer representing the local feature width in pixels.
            You can assume that feature_width will be a multiple of 4 (i.e. every
                cell of your local SIFT-like feature will have an integer width
                and height). This is the initial window size we examine around
                each keypoint.

    Returns:
    -   fv: A numpy array of shape (feat_dim) representing a feature vector.
            "feat_dim" is the feature_dimensionality (e.g. 128 for standard SIFT).
            These are the computed features.

    A useful function to look at would be np.histogram.
    """

    fv = np.zeros(128)
    cellSize = feature_width // 4

    x = int(x)
    y = int(y)

    windowO = orientations[y - 2 * cellSize: y + 2 * cellSize, x - 2 * cellSize: x + 2 * cellSize]
    windowM = magnitudes[y - 2 * cellSize: y + 2 * cellSize, x - 2 * cellSize: x + 2 * cellSize]

    # print(windowO)
    # print(windowM)

    for i in range(cellSize):
        for j in range(cellSize):
            cellO = windowO[i * cellSize: (i + 1) * cellSize, j * cellSize: (j + 1) * cellSize]
            cellM = windowM[i * cellSize: (i + 1) * cellSize, j * cellSize: (j + 1) * cellSize]
            h, _ = np.histogram(cellO, 8, range=(-np.pi, np.pi), weights=cellM)
            # print(cellO)
            # print(cellM)
            # print(h)
            index = i * 32 + j * 8
            fv[index: index + 8] = h

    # print(fv)

    fv /= np.linalg.norm(fv)
    # print(fv)
    return (fv ** 0.9)


def get_features(image, x, y, feature_width):
    """
    This function returns the SIFT features computed at each of the input points
    You should code the above helper functions first, and use them below.
    You should also use your implementation of image gradients from before. 

    Args:
    -   image: A numpy array of shape (m,n), the image
    -   x: A numpy array of shape (k,), the x-coordinates of interest points
    -   y: A numpy array of shape (k,), the y-coordinates of interest points
    -   feature_width: integer representing the local feature width in pixels.
            You can assume that feature_width will be a multiple of 4 (i.e. every
                cell of your local SIFT-like feature will have an integer width
                and height). This is the initial window size we examine around
                each keypoint.

    Returns:
    -   fvs: A numpy array of shape (k, feat_dim) representing all feature vectors.
            "feat_dim" is the feature_dimensionality (e.g. 128 for standard SIFT).
            These are the computed features.
    """
    assert image.ndim == 2, 'Image must be grayscale'

    # print(x.shape[0])
    fvs = np.zeros((x.shape[0], 128))
    ix, iy = get_gradients(image)
    m, o = get_magnitudes_and_orientations(ix, iy)

    # print(m, o)

    i = 0
    for px, py in zip(x, y):
        # print(px, py)
        fvs[i, :] = get_feat_vec(px, py, m, o, feature_width)
        # print(fvs)
        i += 1

    return fvs

