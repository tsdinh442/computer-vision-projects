import numpy as np
import cv2

# define input prompts list
points = []  # list of tuples (x, y) pixel coordinates, need at least 2 points
# define a color
COLOR = (70, 3, 252)
COLOR = (3, 207, 252)

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
                frame, _ = action(frame, points)
                #frame, _ = masking(masks, frame, COLOR, opacity=0.5)

        #mark_dots(masked_image, np.array(points).astype(int))
        cv2.imshow("frame", frame)


def mark_dots(image, points, color=COLOR):
    """

    :param image:
    :param points:
    :param color:
    :return:
    """

    global DOTS_COLOR
    for point in points:
        cv2.circle(image, point, radius=3, color=color, thickness=-1)

    return image


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
        contoured = np.copy(image)
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
            cv2.drawContours(contoured, contours, -1, color, 1)

        return masked_image, binary_mask, contoured


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


def closing_polygon(first_point, last_point):
    """

    :param first_point:
    :param last_point:
    :return:
    """
    thresh = 30
    if np.linalg.norm(np.array(first_point) - np.array(last_point)) < thresh:
        return True
    return False


def display_number_of_cars(image, number_of_cars, font=cv2.FONT_HERSHEY_SIMPLEX):
    """

    :param image:
    :param centroid:
    :param number_of_cars:
    :param font:
    :return:
    """

    X_PADDING = 5
    Y_PADDING = 5

    txt = str(number_of_cars) + ' cars'
    txt_size_x, txt_size_y = cv2.getTextSize(txt, font, fontScale=1, thickness=1)[0]

    x_0 = 30
    y_0 = 30

    txt_position = (x_0 + X_PADDING, y_0 + Y_PADDING + txt_size_y)

    cv2.rectangle(image, (x_0, y_0), ((x_0 + txt_size_x + (X_PADDING * 2)), (y_0 + txt_size_y + (Y_PADDING * 2))), color=(3, 207, 252), thickness=-1)
    cv2.putText(image, txt, txt_position, cv2.FONT_HERSHEY_SIMPLEX, 1, color=(0, 0, 0), thickness=2)