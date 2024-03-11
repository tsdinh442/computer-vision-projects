import cv2
import numpy as np

import sys
sys.path.append('../../1_object_tracking/optical_flow')

from yolov8 import detect
from utils import mark_dots, manhattan_distance



image = cv2.imread('../../media/out/capstone/parking/3.jpg')

_, cars, _, _ = detect(None, image, conf=0.7, iou=0.7)

res, _ = mark_dots(image, cars)

cv2.imwrite('../../media/out/capstone/parking/10.jpg', res)