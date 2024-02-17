import numpy as np
import cv2

MASK_COLOR = (3, 207, 252)
POLYGONS_COLOR = DOTS_COLOR = (70, 3, 252)

# define input prompts list
points = []  # list of tuples (x, y) pixel coordinates, need at least 2 points


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

        mark_dots(frame, np.array(points).astype(int))
        cv2.imshow("frame", frame)


def mark_dots(image, points):
    """

    :param image:
    :param points:
    :return:
    """
    global DOTS_COLOR
    for point in points:
        cv2.circle(image, point, radius=3, color=DOTS_COLOR, thickness=-1)

    return image



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


def out_of_bound(image_shape, points):
    """
    checking if all the points are out of bound
    :param image_shape: tuple (h, w)
    :param points: np array of 2d points
    :return: bool - True if all points are out of bound
    """

    # convert points to integer
    points = np.array(points).astype(int)
    h, w = image_shape
    out_of_bound = ((points[:, 0] < 0) | (points[:, 0] >= w) | (points[:, 1] < 0) | (points[:, 1] >= h))

    return np.all(out_of_bound)
