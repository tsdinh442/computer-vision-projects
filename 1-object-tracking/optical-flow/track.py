import numpy as np
import cv2
from sam import predictor, segment_single_prompt, segment_mult_prompts
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
        labels.append(1)
        # copy the frame
        frame = np.copy(params)
        predictor.set_image(frame)
        if len(points) == 1:
            masks, scores, logits = predictor.predict(
                point_coords=np.array(points),
                point_labels=np.array(labels),
                multimask_output=True,
            )
        elif len(points) > 1:
            mask_input = logits[np.argmax(scores), :, :]  # Choose the model's best mask
            masks, _, _ = predictor.predict(
                point_coords=np.array(points),
                point_labels=np.array(labels),
                mask_input=mask_input[None, :, :],
                multimask_output=False,
            )

        masked_frame = masking(masks, frame, MASK_COLOR, opacity=0.5)
        display_dots(masked_frame, points)

def display_dots(image, points):
    """

    :param image:
    :param points:
    :param color:
    :return:
    """
    global DOTS_COLOR
    for point in points:
        x, y = int(point[0]), int(point[1])
        cv2.circle(image, (x, y), radius=3, color=DOTS_COLOR, thickness=-1)
    cv2.imshow("frame", image)

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

def track(video_path):

    global logits
    global scores
    global MASK_COLOR

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
        #count = 31
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

                if n % 4 == 0:  # compute every 4 frames to reduce the computational cost
                    if len(cur_points) == 1:
                        masks = segment_single_prompt(frame, np.array(cur_points), np.array(labels))
                    elif len(cur_points) > 1:
                        masks = segment_mult_prompts(frame, np.array(cur_points), np.array(labels))

                    #binary_mask = masks[0].astype(np.uint8) * 255
                    masked_frame = masking(masks, frame, MASK_COLOR, opacity=0.5)
                    display_dots(masked_frame, cur_points)

                    # save frames or write into a video
                    #cv2.imwrite("../../media /out2/frames/" + str(count) + ".png", masked_frame)
                    #cv2.imwrite("../../media /out2/masks/" + str(count) + ".png", binary_mask)
                    #count += 1

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
DOTS_COLOR = (70, 3, 252)

# input prompts and labels
points = []  # list of tuples (x, y) pixel coordinates
labels = []  # 1 or 0 - 1 for objects and 0 for background

# run optical flow
track(path)

# destroy and exit
cv2.destroyAllWindows()
