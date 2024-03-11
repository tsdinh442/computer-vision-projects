import pickle
import cv2
import numpy as np
from tensorflow.keras.applications.vgg16 import VGG16
import xgboost as xgb


def classify(image):
    """

    :param vgg:
    :param classifier:
    :param image:
    :param input_size:
    :return:
    """
    input_size = INPUT_SHAPE[:2]
    # resize
    data = cv2.resize(image, input_size)

    # normalize
    data = data / 255

    # expand dim, adding an extra dimension
    input_img = np.expand_dims(data, axis=0)

    # extract features by vgg16
    extracted_features = VGG.predict(input_img)
    print(extracted_features.shape)

    # reshape the feature to the correct shape
    extracted_features = extracted_features.reshape(extracted_features.shape[0], -1)

    # classifying the image
    prediction = XGB.predict(extracted_features)

    return prediction


def label_occupancy(image, mask):
    '''
    predict the number of cars in a video
    input:
    video_path: str, path to the video
    mask_path: str, path to the mask of the frame
    output_folder_path: str, path to the folder where each predicted frame to be saved
    input_size: tuple, the size of the image that the model was trained on
    vgg: pretrained vgg 16
    mode: the model used for prediction

    return None
    '''

    # Define colors in BGR color space
    RED = (0, 0, 255)
    GREEN = (0, 255, 0)

    # Define the text and its properties
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 2

    vacant = []
    occupied = []
    # Find contours in the mask image
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    overlay = image.copy()

    # Iterate over each contour
    for i, contour in enumerate(contours):
        # Create an empty mask for the contour
        contour_mask = np.zeros_like(mask)

        # Draw the current contour on the mask
        cv2.drawContours(contour_mask, [contour], 0, 255, thickness=cv2.FILLED)

        # Bitwise AND operation to crop the image using the contour mask
        extracted_stall = cv2.bitwise_and(image, cv2.cvtColor(contour_mask, cv2.COLOR_GRAY2BGR))

        # Find the bounding rectangle for the contour
        x, y, w, h = cv2.boundingRect(contour)

        # Crop the region from the original image using the bounding rectangle
        cropped = extracted_stall[y:y + h, x:x + w]
        parking_stall = cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR)

        predicted_label = classify(parking_stall)

        if predicted_label == 0:
            vacant.append(contour)
        else:
            occupied.append(contour)

    # color the stall with according colors
    image = cv2.drawContours(image, vacant, -1, GREEN, thickness=cv2.FILLED)
    image = cv2.drawContours(image, occupied, -1, RED, thickness=cv2.FILLED)

    alpha = 0.6
    result = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    # Put the text on the frame
    cv2.rectangle(result, (50, 100), (50 + 150, 100 - 20), (255, 255, 255), thickness=cv2.FILLED)
    cv2.putText(result, 'Available: ' + str(len(vacant)), (50 + 10, 100 - 7), font, font_scale, (0, 0, 0),
                thickness)
    cv2.rectangle(result, (50, 130), (50 + 150, 130 - 20), (255, 255, 255), thickness=cv2.FILLED)
    cv2.putText(result, 'Occupied: ' + str(len(occupied)), (50 + 10, 130 - 7), font, font_scale, (0, 0, 0),
                thickness)

    return result


if __name__ == "__main__":

    INPUT_SHAPE = (32, 32, 3)

    # Load model without classifier/fully connected layers
    VGG = VGG16(weights='imagenet', include_top=False, input_shape=INPUT_SHAPE)

    # Load XGBoost
    with open('./models/XGBoost.pkl', 'rb') as file:
        XGB = pickle.load(file)

    # Make loaded layers as non-trainable. This is important as we want to work with pre-trained weights
    for layer in VGG.layers:
        layer.trainable = False

    ###############

    image_path = "../../media/out/capstone/parking/4.jpg"
    mask_path =  "../../media/collection/satellite/EPSG32614_Date20230928_Lat30.568177_Lon-97.654008_Mpp0.075_mask_2.png"

    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    result = label_occupancy(image, mask)

    cv2.imwrite('../../media/out/capstone/parking/13.jpg', result)
