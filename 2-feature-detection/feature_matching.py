import cv2
import numpy as np


def SIFT(images):

    """

    :param images: list - np arrays of cv2 images
    :return: tuple - (list of np array pf key points, list of np array of their descriptors)
    """
    key_points, descriptors = [], []

    # instantiate sift
    sift = cv2.SIFT_create()

    for image in images:
        kp, des = sift.detectAndCompute(image, None)

        key_points.append(kp)
        descriptors.append(des)

    return key_points, descriptors


def ORB(images):
    """

    :param images:
    :return:
    """

    key_points, descriptors = [], []

    # instantiate orb
    orb = cv2.ORB_create()

    for image in images:
        kp, des = orb.detectAndCompute(image, None)

        key_points.append(kp)
        descriptors.append(des)

    return key_points, descriptors


def generic_matcher(img1, img2, kp_detector):
    """

    :param img1:
    :param img2:
    :param kp_detector:
    :return:
    """

    detect = kp_detector([img1, img2])
    kp1, kp2 = detect[0]
    des1, des2 = detect[1]

    # RootSIFT: Normalize SIFT descriptors and convert to binary
    # des1 /= (np.linalg.norm(des1, axis=1, keepdims=True) + 1e-7)
    #des2 /= (np.linalg.norm(des2, axis=1, keepdims=True) + 1e-7)
    #des1 = np.sqrt(des1)
    #des2 = np.sqrt(des2)
    #des1 = np.uint8(des1 * 255)
    #des2 = np.uint8(des2 * 255)

    # create matcher
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)

    matches = matcher.match(des1, des2, None)

    matches = sorted(matches, key=lambda x: x.distance)

    # draw matches
    matches_image = cv2.drawMatches(img1, kp1, img2, kp2, matches[:5], None,
                                    flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

    return matches_image


def flann_matcher(img1, img2, kp_detector):
    """

    :param img1:
    :param img2:
    :param kp_detector:
    :return:
    """
    # Detect keypoints and compute descriptors

    detect = kp_detector([img1, img2])
    kp1, kp2 = detect[0]
    des1, des2 = detect[1]

    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    # Create FLANN matcher
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Match descriptors
    matches = flann.knnMatch(des1, des2, k=2)

    # Apply ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.5 * n.distance:
            print(m.distance, n.distance)
            good_matches.append(m)

    good_matches = sorted(good_matches, key=lambda x: x.distance)

    # Draw matches
    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, good_matches[:30], None,
                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    return img_matches


def bf_matcher(img1, img2, kp_detector):
    """

    :param img1:
    :param img2:
    :param kp_detector:
    :return:
    """
    detect = kp_detector([img1, img2])
    # Find keypoints and descriptors for both images
    kp1, kp2 = detect[0]
    des1, des2 = detect[1]

    # Create a BFMatcher object
    bf = cv2.BFMatcher()

    # Match descriptors using KNN
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.6 * n.distance:
            good_matches.append(m)

    good_matches = sorted(good_matches, key=lambda x: x.distance)
    # Draw matches
    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, good_matches[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    return img_matches

if __name__ == "__main__":

    path1 = "../media/collection/dji1.png"
    path2 = "../media/collection/satellite.jpg"

    img1 = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)
    img1 = cv2.resize(img1, (0, 0), fx=0.25, fy=0.25)
    img2 = cv2.imread(path2, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.resize(img2, (0, 0), fx=0.25, fy=0.25)

    #match = generic_matcher(img1, img2, kp_detector=ORB)
    #match = flann_matcher(img1, img2, kp_detector=SIFT)
    match = bf_matcher(img1, img2, kp_detector=SIFT)

    cv2.imshow('out', match)
    cv2.imwrite('../media/out/feature_matching/test2.jpg', match)
    cv2.waitKey(0)

    cv2.destroyAllWindows()

