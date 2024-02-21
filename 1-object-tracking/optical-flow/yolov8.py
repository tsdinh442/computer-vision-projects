import cv2
import numpy as np
import torch

from ultralytics import YOLO
from utils import masking, mark_dots, closing_polygon, display_number_of_cars, COLOR

def yolo(model):
    """

    :param model:
    :return:
    """
    # for MacOS
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    # for windows
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'

    car_detector = YOLO(model)
    car_detector.to(device=device)

    return car_detector


def detect(model, targeted_regions, image, conf, iou):

    """

    Parameters
    ----------
    model: YOLO model
    targeted_regions: list - list of vertices
    image: np array - image
    conf: float - confidential threshold
    iou: float - iou threshold

    Returns number of cars (int), list of centroids, list of bboxes, list of conf scores
    -------

    """

    num_of_cars = 0
    centroids = []
    bboxes = []
    scores = []

    # perform the detection task
    detections = model(image, iou=iou)[0]

    for detection in detections.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = detection
        bboxes.append([x1, y1, x2, y2])
        scores.append(score)
        if score > conf:
            centroid = (int((x1 + x2) / 2), int((y1 + y2) / 2))

            # test if car inside the marked parking lot region
            if targeted_regions is not None and len(targeted_regions) > 0:
                check = cv2.pointPolygonTest(targeted_regions, centroid, measureDist=False)
            elif not targeted_regions:
                check = True
            if check > 0:
                num_of_cars += 1
                centroids.append(centroid)

    return num_of_cars, centroids, bboxes, scores


def count_cars(image, points):
    """
    create a mask with boolean value the same size as the input image from the input points
    :param image:
    :param points:
    :return:
    """
    global POLYGONS_COLOR
    global COLOR

    # create a 2d blank black canvas same dimension as the input image
    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    # convert input points to integers
    int_points = np.array(points).astype(int)

    # draw a polygon from input points and fill it with white

    cv2.fillPoly(mask, [int_points.reshape((-1, 1, 2))], color=255)

    # covert binary to boolean matrix
    bool_mask = mask != 0

    masked_image, binary_mask, _ = masking([bool_mask], image, COLOR, opacity=0.25)
    mark_dots(masked_image, int_points)

    # perform car detection to retrieve number of cars and the centroid of each bbox
    if len(points) > 2 and closing_polygon(points[0], points[-1]):
        number_of_cars, centroids, _, _ = detect(car_detector, int_points, image, conf=0.7, iou=0.7)
        mark_dots(masked_image, np.array(centroids).astype(int), color=(0, 0, 255))
        display_number_of_cars(masked_image, number_of_cars)

    return masked_image, binary_mask

##############
car_detector = yolo("../../models/yolo/car_detection.pt")