import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import maximum_filter
import pdb
from itertools import product


def get_gaussian_kernel(ksize, sigma):
    """
    Generate a Gaussian kernel to be used in get_interest_points for calculating
    image gradients and a second moment matrix.
    You can call this function to get the 2D gaussian filter.
    
    This might be useful:
    2) Make sure the value sum to 1
    3) Some useful functions: cv2.getGaussianKernel

    Args:
    -   ksize: kernel size
    -   sigma: kernel standard deviation

    Returns:
    -   kernel: numpy nd-array of size [ksize, ksize]
    """
    
    kernel = cv2.getGaussianKernel(ksize, sigma)
    kernel = np.dot(kernel, kernel.T)
    return kernel

def my_filter2D(image, filt, bias = 0):
    """
    Compute a 2D convolution. Pad the border of the image using 0s.
    Any type of automatic convolution is not allowed (i.e. np.convolve, cv2.filter2D, etc.)

    Helpful functions: cv2.copyMakeBorder

    Args:
    -   image: A numpy array of shape (m,n,c),
                image may be grayscale of color (your choice)
    -   filt: filter that will be used in the convolution

    Returns:
    -   conv_image: image resulting from the convolution with the filter
    """

    m, n, c = image.shape
    image = cv2.copyMakeBorder(image, 1, 1, 1, 1, cv2.BORDER_CONSTANT)
    image = image.reshape((m + 2, n + 2, c))
    conv_image = np.zeros((m - filt.shape[0] + 3, n - filt.shape[1] + 3, c))
    # print(conv_image.shape)
    # print(product(range(conv_image.shape[0]), range(conv_image.shape[1]), range(c)))
    for i, j, k in product(range(conv_image.shape[0]), range(conv_image.shape[1]), range(c)):
        # try:
        conv_image[i, j, k] = np.sum(image[i : i + filt.shape[0], j : j + filt.shape[1], k] * filt) + bias
        # except Exception as e:
        #     print(i, j, k, e)
        #     break

    return conv_image

def get_gradients(image):
    """
    Compute smoothed gradients Ix & Iy. This will be done using a sobel filter.
    Sobel filters can be used to approximate the image gradient
    
    Helpful functions: my_filter2D from above
    
    Args:
    -   image: A numpy array of shape (m,n) containing the image
               

    Returns:
    -   ix: numpy nd-array of shape (m,n) containing the image convolved with differentiated kernel in the x direction
    -   iy: numpy nd-array of shape (m,n) containing the image convolved with differentiated kernel in the y direction
    """
    
    m, n = image.shape
    image = np.reshape(image, (m, n, 1))
    # print(image.shape)
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    ix = np.reshape(my_filter2D(image, sobel_x), (m, n))
    iy = np.reshape(my_filter2D(image, sobel_y), (m, n))
    
    return ix, iy


def remove_border_vals(image, x, y, c, window_size = 16):
    """
    Remove interest points that are too close to a border to allow SIFTfeature
    extraction. Make sure you remove all points where a window around
    that point cannot be formed.

    Args:
    -   image: image: A numpy array of shape (m,n,c),
        image may be grayscale of color (your choice)
    -   x: A numpy array of shape (N,) containing the x coordinate of each pixel
    -   y: A numpy array of shape (N,) containing the y coordinate of each pixel
    -   c: A numpy array of shape (N,) containing the confidences of each pixel
    -   window_size: int of the window size that we want to remove. (i.e. make sure all
        points in a window_size by window_size area can be formed around a point)
        (set this to 16 for unit testing). treat the center point of this window as the bottom right
        of the center most 4 pixels. This will be the same window used for SIFT.

    Returns:
    -   x: A numpy array of shape (N-border_vals_removed,) containing x-coordinates of interest points
    -   y: A numpy array of shape (N-border_vals_removed,) containing y-coordinates of interest points
    -   c: numpy array of shape (N-border_vals_removed,) containing the confidences of each pixel
    """

    blx = (window_size - 2) // 2 + 1
    brx = image.shape[1] - blx
    buy = blx
    bdy = image.shape[0] - buy - 1
    coords = np.reshape(np.dstack((x, y, c)), (-1, 3))
    # print(coords.shape)
    coords = coords[coords[:, 0] >= buy]
    coords = coords[coords[:, 0] <= bdy]
    coords = coords[coords[:, 1] >= blx]
    coords = coords[coords[:, 1] <= brx]
    x, y, c = coords[:, 0], coords[:, 1], coords[:, 2]
    return x, y, c

def second_moments(ix, iy, ksize = 7, sigma = 10):
    """
    Given image gradients, ix and iy, compute sx2, sxsy, sy2 using a gaussian filter.

    Helpful functions: my_filter2D

    Args:
    -   ix: numpy nd-array of shape (m,n) containing the gradient of the image with respect to x
    -   iy: numpy nd-array of shape (m,n) containing the gradient of the image with respect to y
    -   ksize: size of gaussian filter (set this to 7 for unit testing)
    -   sigma: deviation of gaussian filter (set this to 10 for unit testing)

    Returns:
    -   sx2: A numpy nd-array of shape (m,n) containing the second moment in the x direction twice
    -   sy2: A numpy nd-array of shape (m,n) containing the second moment in the y direction twice
    -   sxsy: A numpy nd-array of dim (m,n) containing the second moment in the x then the y direction
    """

    if ksize == 1:
        return ix**2, iy**2, ix*iy

    gk = get_gaussian_kernel(ksize, sigma)
    # print(gk)
    m, n = ix.shape
    sx2 = my_filter2D(np.reshape(ix**2, (m, n, 1)), gk)
    sxsy = my_filter2D(np.reshape(ix*iy, (m, n, 1)), gk)
    sy2 = my_filter2D(np.reshape(iy**2, (m, n, 1)), gk)

    sx2 = np.reshape(sx2, (sx2.shape[0], sx2.shape[1]))
    sxsy = np.reshape(sxsy, (sxsy.shape[0], sxsy.shape[1]))
    sy2 = np.reshape(sy2, (sy2.shape[0], sy2.shape[1]))

    # print(sx2)
    # print(sxsy)
    # print(sy2)

    return sx2, sy2, sxsy

def corner_response(sx2, sy2, sxsy, alpha=0.05):

    """
    Given second moments calculate corner resposne.
    R = det(M) - alpha(trace(M)^2)
    where M = [[Sx2, SxSy],
                [SxSy, Sy2]]


    Args:
    -   sx2: A numpy nd-array of shape (m,n) containing the second moment in the x direction twice
    -   sy2: A numpy nd-array of shape (m,n) containing the second moment in the y direction twice
    -   sxsy: numpy nd-array of dim (m,n) containing the second moment in the x then the y direction
    -   alpha: empirical constant in Corner Resposne equaiton (set this to 0.05 for unit testing)

    Returns:
    -   R: Corner response score for each pixel
    """

    # M = np.vstack((np.hstack((sx2, sxsy)), np.hstack((sxsy, sy2))))
    det = sx2 * sy2 - sxsy ** 2
    trace = sx2 + sy2
    R = det - alpha * trace ** 2

    return R

def non_max_suppression(R, neighborhood_size = 7):
    """
    Implement non maxima suppression. Take a matrix and return a matrix of the same size
    but only the max values in a neighborhood are non zero. We also do not want local
    maxima that are very small as well so remove all values that are below the median.

    Helpful functions: scipy.ndimage.filters.maximum_filter
    
    Args:
    -   R: numpy nd-array of shape (m, n)
    -   ksize: int that is the size of neighborhood to find local maxima (set this to 7 for unit testing)

    Returns:
    -   R_local_pts: numpy nd-array of shape (m, n) where only local maxima are non-zero 
    """

    R[R < np.median(R)] = 0
    # R_max = maximum_filter(R, neighborhood_size)
    # m, n = R.shape
    # for i in range(m):
    #     for j in range(n):
    #         if R[i][j] != R_max[i][j]:
    #             R[i][j] = 0
    return R    

def get_interest_points(image, n_pts = 1500):
    """
    Implement the Harris corner detector (See Szeliski 4.1.1) to start with.
    

    If you're finding spurious interest point detections near the boundaries,
    it is safe to simply suppress the gradients / corners near the edges of
    the image.

    Useful in this function in order to (a) suppress boundary interest
    points (where a feature wouldn't fit entirely in the image, anyway)
    or (b) scale the image filters being used. Or you can ignore it.

    By default you do not need to make scale and orientation invariant
    local features.

    The lecture slides and textbook are a bit vague on how to do the
    non-maximum suppression once you've thresholded the cornerness score.
    You are free to experiment. For example, you could compute connected
    components and take the maximum value within each component.
    Alternatively, you could run a max() operator on each sliding window. You
    could use this to ensure that every interest point is at a local maximum
    of cornerness.

    Args:
    -   image: A numpy array of shape (m,n,c),
                image may be grayscale of color (your choice)
    -   n_pts: integer of number of interest points to obtain

    Returns:
    -   x: A numpy array of shape (n_pts) containing x-coordinates of interest points
    -   y: A numpy array of shape (n_pts) containing y-coordinates of interest points
    -   R_local_pts: A numpy array of shape (m,n) containing cornerness response scores after
            non-maxima suppression and before removal of border scores
    -   confidences: numpy nd-array of dim (n_pts) containing the strength
            of each interest point
    """
    x, y, confidences = [], [], []


    ix, iy = get_gradients(image)
    sx2, sy2, sxsy = second_moments(ix, iy)
    R = corner_response(sx2, sy2, sxsy)
    # print(R)
    R_local_pts = non_max_suppression(R)
    # print(R_local_pts)

    for r, row in enumerate(R_local_pts):
        for c, v in enumerate(row):
            if v > 0:
                y.append(r)
                x.append(c)
                confidences.append(v)

    sort_zip = sorted(zip(confidences, x, y), key = lambda t: t[0], reverse=True)
    x = [a for _, a, _ in sort_zip]
    y = [b for _, _, b in sort_zip]
    confidences = [c for c, _, _ in sort_zip]

    x, y, c = remove_border_vals(image, np.array(x), np.array(y), np.array(confidences))
    
    return x[:n_pts], y[:n_pts], R_local_pts, c[:n_pts]


