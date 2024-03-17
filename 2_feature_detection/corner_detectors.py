import cv2
import numpy as np


def harris_corner(image, quality_level=0.05):

    """
    find corners using harris method
    :param image: np array, cv2 image
    :param quality_level: float - (0, 1) acts as threshold
    :return: tuple (np array - harris corner, np array - overlaid image)
    """

    # convert to gray
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # convert to float
    gray = np.float32(gray)

    # define params
    params = dict(blockSize=5,
                  ksize=3,
                  k=0.04)

    # compute harris corners
    harris = cv2.cornerHarris(gray, **params)

    # select best points and overlay on rgb
    image[harris > quality_level * harris.max()] = [0, 255, 0] # green color

    return harris, image


def good_corners(image, detector='useHarrisDetector'):
    """
    detect good corners using harris or shi tomasi
    :param image: np array - cv2 image
    :param detector: str - default "harris",
    :return: 
    """

    # convert to gray
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # define params
    params = dict(maxCorners=200,
                  qualityLevel=0.5,
                  minDistance=20,
                  blockSize=7,
                  useHarrisDetector=detector == 'useHarrisDetector')

    corners = cv2.goodFeaturesToTrack(gray, **params)

    for corner in corners:
        x, y = corner.ravel().astype(int)
        cv2.circle(image, (x, y), radius=3, color=(0, 0, 255), thickness=-1)

    return corners, image


if __name__ == '__main__':

    # read image
    img_path = "../media/collection/parkinglot2.jpg"
    image = cv2.imread(img_path)

    # harris corners
    harris_corners, harris_image = harris_corner(np.copy(image))

    # good corners
    good_corners, good_image = good_corners(np.copy(image))

    cv2.imshow("harris", harris_image)
    cv2.imshow("good", good_image)
    cv2.waitKey(0)

    cv2.destroyAllWindows()