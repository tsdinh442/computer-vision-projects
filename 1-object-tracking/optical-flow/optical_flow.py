import cv2

def lukas_karnade(prev_gray, cur_gray, prev_points):
    """
    tracking a set of points using lukas karnade method
    :param prev_gray:
    :param cur_gray:
    :param prev_points:
    :return:
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
