import sys

sys.path.append('../optical_flow')

import cv2
import numpy as np

from utils import manhattan_distance, out_of_bound


class Kalman_Filter:

    def __init__(self, coord_x, coord_y):

        stateDim = 4
        measurementDim = 2
        dt = 1
        # self.idx = idx
        self.time = 0
        # self.tracked_point = None
        self.track = True
        self.tracked_point = False
        self.bbox = None

        self.kf = cv2.KalmanFilter(stateDim, measurementDim)  # state vector [x, y, vx, vy] measurement vector [x, y]

        # initialize the kf with starting points
        self.kf.statePost = np.array([coord_x, coord_y, 0, 0], dtype=np.float32)

        # define the transition matrix such that it follows the linear system: _t = x_t-1 + v_xy * dt
        self.kf.transitionMatrix = np.array([[1, 0, dt, 0],
                                        [0, 1, 0, dt],
                                        [0, 0, 1, 0],
                                        [0, 0, 0, 1]], dtype=np.float32)

        self.kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                         [0, 1, 0, 0]], np.float32)

        self.kf.measurementNoiseCov = np.array([[1, 0],
                                           [0, 1]], dtype=np.float32) * 0.05

        self.kf.processNoiseCov = np.array([[1, 0, 0, 0],
                                       [0, 1, 0, 0],
                                       [0, 0, 1, 0],
                                       [0, 0, 0, 1]], np.float32) * 0.03

    def Correct(self, m):
        self.kf.correct(m)
        self.tracked_point = True

    def Predict(self):

        if self.track:
            return self.kf.predict()
        return None

    def Estimate(self):

        return self.kf.statePost

    def Track(self, time):

        self.track = self.time < time

    def Check(self):

        if not self.tracked_point:
            self.time += 1
            self.Track(20)

    def Refresh(self):

        self.tracked_point = False

    def Bbox(self, bbox):

        x1, y1, x2, y2 = bbox
        w = abs(x1 - x2)
        h = abs(y1 - y2)

        center_x, center_y = self.kf.predict()[:2].transpose()[0]
        p1 = int(center_x - w / 2), int(center_y - h / 2)
        p2 = int(center_x + w / 2), int(center_y + h / 2)

        self.bbox = p1, p2
