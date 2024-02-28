import cv2

def lucas_kanade(prev_gray, cur_gray, prev_points):
    """
    track a set of points using lukas kanade method
    :param prev_gray: np array - gray image of previous frame
    :param cur_gray: np array - gray image of current frame
    :param prev_points: np array - set of tracked points in previous frame
    :return: np array - set of estimated tracked points in current frame
    """

    global masks
    global logits
    global scores
    global color

    # Lukas Kanade params
    lk_params = dict(winSize=(10, 10),
                     maxLevel=3,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    cur_points, status, error = cv2.calcOpticalFlowPyrLK(prev_gray, cur_gray, prev_points, None, **lk_params)

    return cur_points
