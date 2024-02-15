import numpy as np
import cv2
from sam import mask_generator, predictor

masks, scores, logits, points = None, None, None, None

path = None
cap = cv2.VidepCapture(path)

while ret:
    ret, frame = cap.read()
    if len(points) > 0:
        for point in points:
            cv2.circle(frame, point, radius=3, color=(0, 255, 0), thickness=-1)
    key = cv2.waitKey(1)
    if key == 27:
        break

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
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
def segment(image, input_points, input_labels):

    global masks
    global scores
    global logits

    input_point = np.array(input_points)
    input_label = np.array(input_labels)

    predictor.set_image(image)

    if len(input_point) == 1:
        masks, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )

    elif len(input_point) > 1:
        mask_input = logits[np.argmax(scores), :, :]  # Choose the model's best mask
        masks, _, _ = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            mask_input=mask_input[None, :, :],
            multimask_output=False,
        )

    return masks
