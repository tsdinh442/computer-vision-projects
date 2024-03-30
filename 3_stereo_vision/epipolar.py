import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

class Epipolar:

    def __init__(self, left_path, right_path):

        self.image_left = cv2.imread(left_path, 0)
        self.image_right = cv2.imread(right_path, 0)

        self.ptsLeft = None
        self.ptsRight = None

        self.F = None

        self._get_fundamental_matrix()

    def _get_keypoints(self):

        sift = cv2.SIFT_create()
        kpLeft, desLeft = sift.detectAndCompute(self.image_left, None)
        kpRight, desRight = sift.detectAndCompute(self.image_right, None)
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(desLeft, desRight, k=2)

        ptsLeft, ptsRight = [], []
        for i , (m, n) in enumerate(matches):
            if m.distance < 0.8 * n.distance:
                ptsLeft.append(kpLeft[m.queryIdx].pt)
                ptsRight.append(kpRight[m.trainIdx].pt)

        ptsLeft = np.int32(ptsLeft)
        ptsRight = np.int32(ptsRight)

        return ptsLeft, ptsRight

    def _get_fundamental_matrix(self):

        ptsLeft, ptsRight = self._get_keypoints()

        self.F, mask = cv2.findFundamentalMat(ptsLeft, ptsRight, cv2.FM_LMEDS)

        self.ptsLeft = ptsLeft[mask.ravel() == 1][::5, :]
        self.ptsRight = ptsRight[mask.ravel() == 1][::5, :]

        self.colors = [tuple(np.random.randint(0, 255, 3).tolist()) for i in range(len(self.ptsRight))]

    def _get_epilines(self, pts, whichImage):

        lines = cv2.computeCorrespondEpilines(pts.reshape(-1, 1, 2), whichImage, self.F)
        return lines.reshape(-1, 3)

    def _draw_epilines(self, pts1, pts2, lines, pt_image, line_image):

        h, w = line_image.shape

        pt_image = cv2.cvtColor(pt_image, cv2.COLOR_GRAY2BGR)
        line_image = cv2.cvtColor(line_image, cv2.COLOR_GRAY2BGR)

        for l, pt1, pt2, color in zip(lines, pts1, pts2, self.colors):
            start_x, start_y = map(int, [0, -l[2] / l[1]])
            end_x, end_y = map(int, [w, -(l[2] + l[0] * w) / l[1]])

            line_image = cv2.line(line_image, (start_x, start_y), (end_x, end_y), color=color, thickness=3)
            line_image = cv2.circle(line_image, pt2, radius=15, color=color, thickness=-1)
            pt_image = cv2.circle(pt_image, pt1, radius=15, color=color, thickness=-1)

        return pt_image, line_image

    def draw_epilines_left(self):
        lines = self._get_epilines(self.ptsRight, 2)
        _, image_left = self._draw_epilines(self.ptsRight, self.ptsLeft, lines, self.image_right, self.image_left)

        return image_left
        #self.show_images(image_left, image_right)

    def draw_epilines_right(self):
        lines = self._get_epilines(self.ptsLeft, 1)
        _, image_right = self._draw_epilines(self.ptsLeft, self.ptsRight, lines, self.image_left, self.image_right)
        return image_right
        #self.show_images(image_left, image_right)


    def show_epilines(self):

        #color = tuple(np.random.randint(0, 255, 3).tolist())

        image_left = self.draw_epilines_left()
        image_right = self.draw_epilines_right()

        plt.figure()
        plt.subplot(121)
        plt.imshow(image_left, cmap='gray')
        plt.subplot(122)
        plt.imshow(image_right, cmap='gray')

        plt.show()




if __name__ == '__main__':

    img_left = '../media/collection/stereo/stereo_left.jpg'
    img_right = '../media/collection/stereo/stereo_right.jpg'

    epipolar = Epipolar(img_left, img_right)
    epipolar.show_epilines()

