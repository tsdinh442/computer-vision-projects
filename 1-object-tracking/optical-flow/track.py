import numpy as np
import cv2
from sam import segment
from optical_flow import lukas_karnade
from utils import select_points, mark_dots, draw_polygons, masking, out_of_bound, points, MASK_COLOR, POLYGONS_COLOR


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
        count = 31
        while True:

            if key == ord('q'):
                break

            # read the next frame
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            cur_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if len(prev_points) > 0:
                cur_points = lukas_karnade(prev_gray, cur_gray, prev_points)

                # check if point out of bound
                if out_of_bound(frame.shape[:2], cur_points):
                    break

                if n % 3 == 0:  # compute every 4 frames to reduce the computational cost

                    # compute masks - either from sam or draw from points
                    masks = action(frame, cur_points)
                    binary_mask = masks[0].astype(np.uint8) * 255
                    masked_frame = masking(masks, frame, MASK_COLOR, opacity=0.5)
                    mark_dots(masked_frame, cur_points.astype(int))

                    cv2.imshow("frame", masked_frame)

                    # save frames or write into a video
                    cv2.imwrite("../../media /out/frames/" + str(count) + ".png", masked_frame)
                    cv2.imwrite("../../media /out/masks/" + str(count) + ".png", binary_mask)
                    count += 1
                n += 1

                prev_points = cur_points
                prev_gray = cur_gray

    # Release the VideoWriter object and close the output file
    cap.release()

###### main

# media path
path = "../../media /videos/4.mp4"

# global variables
scores = None
logits = None

# run optical flow
track(path, action=draw_polygons)

# destroy and exit
cv2.destroyAllWindows()
