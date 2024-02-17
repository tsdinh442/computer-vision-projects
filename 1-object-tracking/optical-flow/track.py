import numpy as np
import cv2
from sam import predictor, segment_single_prompt, segment_mult_prompts, segment
from optical_flow import lukas_karnade


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
    global scores
    global logits
    global MASK_COLOR

    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))

        # get video frame
        frame = np.copy(params['frame'])

        if len(points) > 0:
            # get the action
            action = params.get('action')
            if callable(action):
                masks = action(frame, points)
                frame = masking(masks, frame, MASK_COLOR, opacity=0.5)

        display_dots(frame, np.array(points).astype(int))


def display_dots(image, points):
    """

    :param image:
    :param points:
    :return:
    """
    global DOTS_COLOR
    for point in points:
        print(point)
        cv2.circle(image, point, radius=3, color=DOTS_COLOR, thickness=-1)

    cv2.imshow("frame", image)


def draw_polygons(image, points):
    """
    create a mask with boolean value the same size as the input image from the input points
    :param image:
    :param points:
    :return:
    """
    global POLYGONS_COLOR

    # create a 2d blank black canvas same dimension as the input image
    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    # convert input points to integers
    int_points = np.array(points).astype(int)

    # draw a polygon from input points and fill it with white
    cv2.fillPoly(mask, [int_points.reshape((-1, 1, 2))], color=255)

    # covert binary to boolean matrix
    bool_mask = mask != 0

    return [bool_mask]


def masking(masks, image, color, opacity=0.35):
    """

    :param masks:
    :param image:
    :param color:
    :param opacity:
    :return:
    """

    if masks is not None:
        image_copy = np.copy(image)
        color_mask = np.copy(image)
        for mask in masks:
            # Convert boolean mask to binary mask
            binary_mask = mask.astype(np.uint8) * 255
            # Convert the integer mask to a 3-channel mask
            color_mask[mask] = color

            # Blend the image and the color mask
            masked_image = cv2.addWeighted(color_mask, opacity, image_copy, 1 - opacity, 0, image_copy)

            # Find contours in the binary mask
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # Draw contours on the masked image
            cv2.drawContours(masked_image, contours, -1, color, 1)

        return masked_image


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

    key = cv2.waitKey(0)

    prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    prev_points = np.array(points).astype(np.float32) if len(points) > 0 else np.array([], dtype=float)

    if key == 13:
        n = 0
        count = 62
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
                print(n, cur_points)
                if n % 4 == 0:  # compute every 4 frames to reduce the computational cost

                    masks = action(frame, cur_points)
                    #binary_mask = masks[0].astype(np.uint8) * 255
                    masked_frame = masking(masks, frame, MASK_COLOR, opacity=0.5)
                    display_dots(masked_frame, cur_points.astype(int))

                    # save frames or write into a video
                    cv2.imwrite("../../media /out2/" + str(count) + ".png", masked_frame)
                    #cv2.imwrite("../../media /out2/masks/" + str(count) + ".png", binary_mask)
                    count += 1

                n += 1

                prev_points = cur_points
                prev_gray = cur_gray


    # Release the VideoWriter object and close the output file
    cap.release()

###### main

# media path
path = "../../media /videos/2.mp4"

# global variables
scores = None
logits = None
MASK_COLOR = (3, 207, 252)
POLYGONS_COLOR = DOTS_COLOR = (70, 3, 252)

# define input prompts list
points = []  # list of tuples (x, y) pixel coordinates, need at least 2 points

# run optical flow
track(path, action=segment)

# destroy and exit
cv2.destroyAllWindows()
