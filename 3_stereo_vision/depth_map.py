import cv2
import numpy as np
import matplotlib.pyplot as plt

class DepthMap:

    def __init__(self, path_left, path_right):
        self.image_left = cv2.imread(path_left, cv2.IMREAD_GRAYSCALE)
        self.image_right = cv2.imread(path_right, cv2.IMREAD_GRAYSCALE)
        self.depth_map = None

    def displayImages(self):

        plt.figure()
        plt.subplot(131)
        plt.imshow(self.image_left)
        plt.subplot(132)
        plt.imshow(self.image_right)
        plt.subplot(133)
        plt.imshow(self.depth_map, 'gray')
        plt.show()

    def computeDepthMap_BM(self):

        disparityFactor = 6
        stereo = cv2.StereoBM.create(numDisparities=16*disparityFactor,
                                     blockSize=21)
        self.depth_map = stereo.compute(self.image_left, self.image_right)

        self.displayImages()


    def computeDepthMap_SGBM(self):
        windowSize = 3
        minDisparity = 0
        disparityFactor = 4
        numDisparities = 16 * disparityFactor

        stereo = cv2.StereoSGBM_create(minDisparity=minDisparity,
                                       numDisparities=numDisparities,
                                       blockSize=windowSize,
                                       P1=8*3*windowSize**2,
                                       P2=32*3*windowSize**2,
                                       #disp12MaxDiff=12,
                                       #uniquenessRatio=10,
                                       #speckleWindowSize=50,
                                       #speckleRange=32,
                                       #preFilterCap=63,
                                       mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)

        self.depth_map = stereo.compute(self.image_left, self.image_right).astype(np.float32) / 16.0


        self.displayImages()



if __name__ == '__main__':

    path_left = '../media/collection/stereo/scene1.row3.col1.ppm'
    path_right = '../media/collection/stereo/scene1.row3.col5.ppm'

    depth_map = DepthMap(path_left, path_right)
    # depth_map.computeDepthMap_BM()
    depth_map.computeDepthMap_SGBM()