import numpy as np
import cv2
from sam import sam

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
    global masks
    global scores
    global logits

    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        labels.append(1)
        # copy the frame
        frame = np.copy(params)
        if len(points) == 1:
            masks, scores, logits = segment(frame, np.array(points), np.array(labels))
        elif len(points) > 1:
            masks, scores, logits = segment(frame, np.array(points), np.array(labels), logits=logits, scores=scores)

        masked_frame = masking(masks, frame, opacity=0.5)

        cv2.circle(masked_frame, (x, y), radius=3, color=(0, 255, 0), thickness=-1)
        cv2.imshow("frame", masked_frame)

def resize(image, scale_factor):
    """

    :param image: image in np array
    :param scale_factor: float scale factor
    :return: resized image in np array
    """
    return cv2.resize(image, (0, 0), fx=scale_factor, fy=scale_factor)

def segment(image, input_points, input_labels, logits=None, scores=None):
    """

    :param image:
    :param input_points:
    :param input_labels:
    :return:
    """

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

    return masks, scores, logits

def masking(masks, image, opacity=0.5):

    if masks is not None:
        masked_image = np.copy(image)
        color_mask = np.zeros_like(image)

        for mask in masks:

            color = np.random.random_integers(0, 255, 3)
            # overlay the mask over an image
            color_mask[mask] = color
            # Add the colored polygon to the original image with opacity
            masked = cv2.addWeighted(color_mask, opacity, masked_image, 1 - opacity, 0, masked_image)
            binary_mask = mask.astype(np.uint8) * 255
            # Find contours in the binary mask
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContour(masked, contours, -1, color, 2)

        return masked

def optical_flow(video_path):

    global masks
    global logits
    global scores

    # Lukas Kanade params
    lk_params = dict(winSize=(10, 10),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    frame = resize(frame, 0.5)
    cv2.imshow("frame", frame)
    # select points on the picture frame
    cv2.setMouseCallback("frame", select_points, param=frame)

    prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    prev_points = np.array(points).astype(np.float32) if len(points) > 0 else np.array([], dtype=float)

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    h, w = frame.shape[:2]
    out = cv2.VideoWriter('output.avi', fourcc, 20.0, (w, h))

    while True:
        # allowing the user to pause the video
        key = cv2.waitKey(33)
        if key == ord('q'):
            break
        elif key == 13:
            cv2.waitKey(0)

        # read the next frame
        ret, frame = cap.read()
        if not ret:
            break
        frame = resize(frame, 0.5)
        cur_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


        if len(prev_points) > 0:
            cur_points, status, error = cv2.calcOpticalFlowPyrLK(prev_gray, cur_gray, prev_points, None, **lk_params)
            print(cur_points)
            prev_gray = cur_gray
            prev_points = cur_points

            if len(cur_points) == 1:
                masks, scores, logits = segment(frame, np.array(points), np.array(labels))
            elif len(cur_points) > 1:
                masks, scores, logits = segment(frame, np.array(points), np.array(labels), logits=logits, scores=scores)

            masked_frame = masking(masks, frame, opacity=0.5)

            for (x, y) in cur_points:
                point = (int(x), int(y))
                cv2.circle(masked_frame, point, radius=3, color=(0, 255, 0), thickness=-1)

            cv2.imshow("frame", masked_frame)
            out.write(masked_frame)

        else:
            prev_points = np.array(points).astype(np.float32)
            cv2.imshow("frame", frame)
            out.write(frame)


        # select points on the picture frame
        cv2.setMouseCallback("frame", select_points, param=frame)

    out.release()
    cap.release()

###### main

# sam
check_point = 'models/sam_vit_h_4b8939.pth'
model_type = 'vit_h'
mask_generator, predictor = sam(check_point, model_type)

# media path
path = "../../media /videos/1.mp4"

# global variables
masks = None
scores = None
logits = None
points = []
labels = []

# run optical flow
optical_flow(path)

# destroy and exit
cv2.destroyAllWindows()
