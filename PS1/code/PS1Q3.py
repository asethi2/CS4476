import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage import io

class Prob3():
    def __init__(self):
        """Load input color image inputPS1Q3.jpg here and assign it as a class variable"""
        self.A = io.imread('./inputPS1Q3.jpg')
    
    def rgb2gray(self, rgb):
        """
        Do RGB to Gray image conversion here. Input is the RGB image and you must return the grayscale image as gray

        Returns:
            gray: grayscale image (1 channeled image with integer values lying between 0 - 255)
        """
        gray = np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])
        # print(gray)
        
        return gray.astype('int64')
        
    def prob_3_1(self):
        """
        Swap red and green color channels here, and return swapImg

        Returns:
            swapImg: RGB image with R and G channels swapped (3 channeled image with integer values lying between 0 - 255)
        """
        swapImg = np.full(self.A.shape, 0)
        swapImg = self.A[:, :, [1, 0, 2]]
        # print(self.A)
        # print(swapImg)
        plt.imshow(swapImg)
        plt.imsave('./PS1Q3prob_3_1.jpg', swapImg)
        plt.show()
    
        return swapImg
    
    def prob_3_2(self):
        """
        This function would simply call your rgb2gray function and return the grayscale image.

        Returns:
            grayImg: grayscale image (1 channeled image with integer values lying between 0 - 255)
        """
        grayImg = self.rgb2gray(self.A)
        plt.imshow(grayImg, cmap='gray')
        plt.imsave('./PS1Q3prob_3_2.jpg', grayImg, cmap='gray')
        plt.show()
    
        return grayImg
    
    def prob_3_3(self):
        """
        Convert grayscale image to its negative.

        Returns:
            negativeImg: negative image (1 channeled image with integer values lying between 0 - 255)
        """
        
        negativeImg = 255 - self.rgb2gray(self.A)
        plt.imshow(negativeImg, cmap='gray')
        plt.imsave('./PS1Q3prob_3_3.jpg', negativeImg, cmap='gray')
        plt.show()
    
        return negativeImg
    
    def prob_3_4(self):
        """
        Create mirror image of gray scale image here.
        
        Returns:
            mirrorImg: mirror image (1 channeled image with integer values lying between 0 - 255)
        """
        
        mirrorImg = self.rgb2gray(self.A)[:, ::-1]
        plt.imshow(mirrorImg, cmap='gray')
        plt.imsave('./PS1Q3prob_3_4.jpg', mirrorImg, cmap='gray')
        plt.show()
    
        return mirrorImg
    
    def prob_3_5(self):
        """
        Average grayscale image with mirror image here.
        
        Returns:
            avgImg: average of grayscale and mirror image (1 channeled image with integer values lying between 0 - 255)
        """

        avgImg = (self.rgb2gray(self.A) + self.rgb2gray(self.A)[:, ::-1]) / 2
        avgImg = avgImg.astype('int')

        np.save('./jeremy.npy', avgImg)

        plt.imshow(avgImg, cmap='gray')
        plt.imsave('./PS1Q3prob_3_5.jpg', avgImg, cmap='gray')
        plt.show()
    
        return avgImg
    
    def prob_3_6(self):
        """
        Create noise matrix N and save as noise.npy. Add N to grayscale image, clip to ensure that max value is 255.
        
        Returns:
            addNoiseImg: grayscale image after adding noise (1 channeled image with integer values lying between 0 - 255)
        """
        
        N = np.load('./noise.npy')

        addNoiseImg = np.clip(self.rgb2gray(self.A) + N, 0, 255)
        addNoiseImg = addNoiseImg.astype('int')
        plt.imshow(addNoiseImg, cmap='gray')
        plt.imsave('./PS1Q3prob_3_6.jpg', addNoiseImg, cmap='gray')
        plt.show()
    
        return addNoiseImg

        
        
if __name__ == '__main__':
    
    p3 = Prob3()
    
    swapImg = p3.prob_3_1()
    grayImg = p3.prob_3_2()
    negativeImg = p3.prob_3_3()
    mirrorImg = p3.prob_3_4()
    avgImg = p3.prob_3_5()
    addNoiseImg = p3.prob_3_6()