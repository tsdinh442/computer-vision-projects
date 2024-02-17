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
    global color

    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        labels.append(1)
        # copy the frame
        frame = np.copy(params)
        if len(points) == 1:
            masks, scores, logits = segment(frame, np.array(points), np.array(labels))
        elif len(points) > 1:
            masks, scores, logits = segment(frame, np.array(points), np.array(labels), logits=logits, scores=scores)

        masked_frame = masking(masks, frame, color, opacity=0.5)

        for point in points:
            cv2.circle(masked_frame, point, radius=3, color=(127, 3, 253), thickness=-1)
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

def masking(masks, image, color, opacity=0.35):

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

            # Create a masked image by combining the overlay and the original image
            # masked_image = cv2.bitwise_and(image_copy, overlay)

            # Find contours in the binary mask
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # Draw contours on the masked image
            cv2.drawContours(masked_image, contours, -1, color, 2)

        return masked_image

def optical_flow(prev_gray, cur_gray, prev_points):

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

def track(video_path):

    global masks
    global logits
    global scores
    global color

    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    cv2.imshow("frame", frame)
    # select points on the picture frame
    cv2.setMouseCallback("frame", select_points, param=frame)
    key = cv2.waitKey(0)

    prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    prev_points = np.array(points).astype(np.float32) if len(points) > 0 else np.array([], dtype=float)

    if key == 13:
        n = 0
        count = 31
        while True:
            # allowing the user to pause the video
            key = cv2.waitKey(33)
            if key == ord('q'):
                break

            # read the next frame
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            cur_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if len(prev_points) > 0:
                cur_points = optical_flow(prev_gray, cur_gray, prev_points)

                if n % 4 == 0:
                    if len(cur_points) == 1:
                        masks, scores, logits = segment(frame, np.array(prev_points), np.array(labels))
                    elif len(cur_points) > 1:
                        for idx, point in enumerate(cur_points):
                            if idx == 0:
                                masks, scores, logits = segment(frame, np.array([point]), np.array([labels[0]]))
                        masks, scores, logits = segment(frame, np.array(cur_points), np.array(labels), logits=logits, scores=scores)

                    binary_mask = masks[0].astype(np.uint8) * 255
                    masked_frame = masking(masks, frame, color, opacity=0.5)

                    for (x, y) in cur_points:
                        point = (int(x), int(y))
                        cv2.circle(masked_frame, point, radius=3, color=(127, 3, 253), thickness=-1)

                    cv2.imshow("frame", masked_frame)
                    cv2.imwrite("../../media /out2/frames/" + str(count) + ".png", masked_frame)
                    cv2.imwrite("../../media /out2/masks/" + str(count) + ".png", binary_mask)
                    count += 1

                n += 1

            prev_points = cur_points
            prev_gray = cur_gray

    # Release the VideoWriter object and close the output file
    cap.release()

###### main

# sam
check_point = 'models/sam_vit_h_4b8939.pth'
model_type = 'vit_h'
mask_generator, predictor = sam(check_point, model_type)

# media path
path = "../../media /videos/2.mp4"
# global variables
masks = None
scores = None
logits = None
points = []
labels = []
color = (198, 252, 3)

# run optical flow
track(path)

# destroy and exit
cv2.destroyAllWindows()
