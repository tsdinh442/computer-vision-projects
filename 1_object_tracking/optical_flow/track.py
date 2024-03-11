import numpy as np
import cv2
# from sam import segment
from optical_flow import lucas_kanade
from utils import select_points, mark_dots, masking, out_of_bound, points
from yolov8 import count_cars


def track(video_path, action=None):

    global logits
    global scores
    global MASK_COLOR

    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    cv2.imshow("frame", frame)

    # select points on the picture frame
    # callback params
    params = {'action': action,
              'frame': frame
              }
    cv2.setMouseCallback("frame", select_points, param=params)

    # pause the video on first frame
    key = cv2.waitKey(0)

    prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    prev_points = np.array(points).astype(np.float32) if len(points) > 0 else np.array([], dtype=float)

    # resume streaming if enter is pressed
    if key == 13:
        n = 0
        print(n)
        count = 81
        while True:

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # read the next frameqq
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            cur_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if len(prev_points) > 0:
                cur_points = lucas_kanade(prev_gray, cur_gray, prev_points)

                # check if point out of bound
                if out_of_bound(frame.shape[:2], cur_points):
                    break

                if n % 3 == 0:  # compute every number of frames to reduce the computational cost

                    # compute masks - either from sam or draw from points
                    result, binary_mask = action(frame, cur_points)
                    cv2.imshow("output", result)

                    # save frames or write into a video
                    cv2.imwrite("../../media/out/frames/" + str(count) + ".png", result)
                    cv2.imwrite("../../media/out/masks/" + str(count) + ".png", binary_mask)
                    count += 1
                n += 1

                prev_points = cur_points
                prev_gray = cur_gray

    # Release the VideoWriter object and close the output file
    cap.release()

###### main

# media path
path = "../../media/videos/5.mp4"

# global variables
scores = None
logits = None
surface_lot = []

MASK_COLOR = (3, 207, 252)

# run optical flow
# track(path, action=count_cars)  # track a surface

track(path, action=count_cars)  # track and segment an object from the background.

# destroy and exit
cv2.destroyAllWindows()
