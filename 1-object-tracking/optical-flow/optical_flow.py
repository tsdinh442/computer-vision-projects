import numpy as np
import cv2
from sam import mask_generator, predictor

def select_points(event, x, y, flags, params):
    """
    mouse callback function
    :param event:
    :param x:
    :param y:
    :param flags:
    :param params:
    :return:
    """
    global points
    global frame
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        cv2.circle(frame, (x, y), radius=3, color=(0, 255, 0), thickness=-1)
        cv2.imshow("frame", frame)

def resize(image, scale_factor):
    """

    :param image: image in np array
    :param scale_factor: float scale factor
    :return: resized image in np array
    """
    return cv2.resize(image, (0, 0), fx=scale_factor, fy=scale_factor)

def segment(image, input_points):
    """

    :param image:
    :param input_points:
    :param input_labels:
    :return:
    """

    global masks
    global scores
    global logits

    input_labels = np.ones((input_points.size, 1))

    predictor.set_image(image)

    if len(input_points) == 1:
        masks, scores, logits = predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            multimask_output=True,
        )

    elif len(input_points) > 1:
        mask_input = logits[np.argmax(scores), :, :]  # Choose the model's best mask
        masks, _, _ = predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            mask_input=mask_input[None, :, :],
            multimask_output=False,
        )

    return masks

def optical_flow(video_path):

    global frame
    # Lukas Kanade params
    lk_params = dict(winSize=(10, 10),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    frame = resize(frame, 0.5)
    cv2.imshow("frame", frame)

    prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    prev_points = np.array(points).astype(np.float32) if len(points) > 0 else np.array([], dtype=float)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = resize(frame, 0.5)

        cur_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_points.size > 0:
            cur_points, status, error = cv2.calcOpticalFlowPyrLK(prev_gray, cur_gray, prev_points, None, **lk_params)
            print(cur_points)
            prev_gray = cur_gray
            prev_points = cur_points
            for (x, y) in cur_points:
                point = (int(x), int(y))
                cv2.circle(frame, point, radius=3, color=(0, 255, 0), thickness=-1)
        else:
            prev_points = np.array(points).astype(np.float32)

        cv2.imshow("frame", frame)
        key = cv2.waitKey(33)
        if key == ord('q'):
            break
        elif key == 13:
            cv2.waitKey(0)

    cap.release()

cv2.namedWindow("frame")
cv2.setMouseCallback("frame", select_points)


###### main
path = "../../media /videos/1.mp4"
masks, scores, logits = None, None, None
points = []
frame = None
optical_flow(path)


cv2.destroyAllWindows()
