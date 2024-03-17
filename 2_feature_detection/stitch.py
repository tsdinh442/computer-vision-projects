import cv2
import numpy as np
import os

from feature_matching import flann_matcher, bf_matcher, generic_matcher, SIFT



def stitch(images_path, satellite_path):
    """
    stitch drone images into one based on the satellite image
    :param images_path: string - path to the directory that contains a list of drone images to be stitched together;
                                images must be named as numeric + extension, and named in the order of how the images were taken
    :param destination_path: string - path to the satellite image
    :return: tuple - stitched image with satellite background, stitched image with black background
    """

    # read source/satellite image
    destination = cv2.imread(satellite_path)
    background = np.zeros_like(destination)
    h, w, _ = destination.shape

    # access and sort the file names
    images = os.listdir(images_path)
    images = [file for file in images if file.lower().endswith('.jpg')]
    images = sorted(images, key=lambda x: int(x.split(".")[0]), reverse=True)

    for image in images[:6]:
        # read source images
        path = os.path.join(images_path, image)
        source = cv2.imread(path)

        # run feature detector discriptor
        _, matches, src_keypoints, des_keypoints = flann_matcher(source, destination, SIFT)

        # extract point coordinates
        src_points = np.float32([src_keypoints[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        des_points = np.float32([des_keypoints[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # compute homography matrix
        err_threshold = 3
        H, mask = cv2.findHomography(src_points, des_points, cv2.RANSAC, err_threshold)
        # matches_mask = mask.ravel().tolist()

        # warp perspective
        warped = cv2.warpPerspective(source, H, (w, h))

        background[warped != 0] = warped[warped != 0]

        # overlay warped image on to the source
        warped[warped == 0] = destination[warped == 0]
        # reassign the destination to be the warped image
        destination = warped

        # define params
        # draw_params = dict(matchColor=(0, 255, 0), singlePointColor=None, matchesMask=matches_mask,
        #                   flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
        # draw matches
        # image_match = cv2.drawMatches(source, src_keypoints, destination, des_keypoints, matches, None, **draw_params)

    return destination, background


if __name__ == "__main__":

    root = '/Users/tungdinh/Desktop/230919_Texas State Roundrock/4 PM'
    satellite_path = "../media/collection/satellite/satellite.png"

    filled_bg, transparent_bg = stitch(root, satellite_path)

    cv2.imwrite(f"../media/out/feature_matching/filled_.jpg", filled_bg)
    cv2.imwrite(f"../media/out/feature_matching/trans_.jpg", transparent_bg)


