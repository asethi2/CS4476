import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage import io

class Prob4():
    def __init__(self):
        """Load input color image indoor.png and outdoor.png here as class variables."""
        self.indoor = io.imread('./indoor.png')
        self.outdoor = io.imread('./outdoor.png')
        # print(self.indoor.shape)
        # print(self.outdoor.shape)
    
    def prob_4_1(self):
        """Plot R,G,B channels separately and also their corresponding LAB space channels separately"""
        indoor_red = self.indoor[:, :, [0]].reshape((self.indoor.shape[0], self.indoor.shape[1]))
        indoor_green = self.indoor[:, :, [1]].reshape((self.indoor.shape[0], self.indoor.shape[1]))
        indoor_blue = self.indoor[:, :, [2]].reshape((self.indoor.shape[0], self.indoor.shape[1]))
        outdoor_red = self.outdoor[:, :, [0]].reshape((self.outdoor.shape[0], self.outdoor.shape[1]))
        outdoor_green = self.outdoor[:, :, [1]].reshape((self.outdoor.shape[0], self.outdoor.shape[1]))
        outdoor_blue = self.outdoor[:, :, [2]].reshape((self.outdoor.shape[0], self.outdoor.shape[1]))
        plt.imsave('./PS1Q4prob_4_1_in_red.png', indoor_red, cmap='gray')
        plt.imsave('./PS1Q4prob_4_1_in_green.png', indoor_green, cmap='gray')
        plt.imsave('./PS1Q4prob_4_1_in_blue.png', indoor_blue, cmap='gray')
        plt.imsave('./PS1Q4prob_4_1_out_red.png', outdoor_red, cmap='gray')
        plt.imsave('./PS1Q4prob_4_1_out_green.png', outdoor_green, cmap='gray')
        plt.imsave('./PS1Q4prob_4_1_out_blue.png', outdoor_blue, cmap='gray')
        plt.imshow(indoor_red, cmap='gray')
        plt.show()
        plt.imshow(indoor_green, cmap='gray')
        plt.show()
        plt.imshow(indoor_blue, cmap='gray')
        plt.show()
        plt.imshow(outdoor_red, cmap='gray')
        plt.show()
        plt.imshow(outdoor_green, cmap='gray')
        plt.show()
        plt.imshow(outdoor_blue, cmap='gray')
        plt.show()

        indoor_lab = cv2.cvtColor(self.indoor, cv2.COLOR_RGB2LAB)
        outdoor_lab = cv2.cvtColor(self.outdoor, cv2.COLOR_RGB2LAB)
        cv2.imwrite('./PS1Q4prob_4_1_indoor_lab.png', indoor_lab)
        cv2.imwrite('./PS1Q4prob_4_1_outdoor_lab.png', outdoor_lab)
        cv2.imshow('indoor_lab',indoor_lab)
        cv2.imshow('outdoor_lab',outdoor_lab)
        cv2.waitKey(5000)


    def prob_4_3(self):
        """
        Convert the loaded RGB image to HSV and return HSV matrix without using inbuilt functions. Return the HSV image as HSV. Make sure to use a 3 channeled RGB image with floating point values lying between 0 - 1 for the conversion to HSV.

        Returns:
            HSV image (3 channeled image with floating point values lying between 0 - 1 in each channel)
        """
        
        img = io.imread('inputPS1Q4.jpg') 
        img = img / 255.0
        # print(img.shape)
        R = img[:, :, 0]
        G = img[:, :, 1]
        B = img[:, :, 2]
        V = np.max(img, axis=2)
        print(V.dtype)
        mask1 = V == 0
        print(len(mask1))
        print(mask1[mask1])
        mask2 = V != 0
        m = np.min(img, axis=2)
        C = V - m
        print(C.dtype)
        print(len(C == 0))
        S = np.copy(C)
        # print(S.shape)
        S[mask1] = 0
        S[mask2] = C[mask2] / V[mask2]
        print(len(S == 0))
        _H = np.copy(C)
        maskR = V == R
        maskR &= mask2
        maskG = V == G
        maskG &= mask2
        maskB = V == B
        maskB &= mask2
        _H[maskR] = (G[maskR] - B[maskR]) / C[maskR]
        _H[maskG] = (B[maskG] - R[maskG]) / C[maskG] + 2
        _H[maskB] = (R[maskB] - G[maskB]) / C[maskB] + 4

        H = _H / 6
        H[H < 0] += 1

        HSV = np.concatenate((np.expand_dims(H, axis=2), np.expand_dims(S, axis=2), np.expand_dims(V, axis=2)), axis = 2)
        HSV = np.nan_to_num(HSV)

        plt.imsave('./outputPS1Q4.png', HSV, cmap='hsv')
        plt.imshow(HSV, cmap='hsv')

        return HSV
        

        
if __name__ == '__main__':
    
    p4 = Prob4()
    
    # p4.prob_4_1()

    HSV = p4.prob_4_3()





