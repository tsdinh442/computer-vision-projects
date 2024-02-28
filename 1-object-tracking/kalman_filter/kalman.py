import cv2
import numpy as np


def kalman_filter():

    stateDim = 4
    measurementDim = 2
    dt = 1

    kf = cv2.KalmanFilter(stateDim, measurementDim)  # state vector [x, y, vx, vy] measurement vector [x, y]

    # define the transition matrix such that it follows the linear system: _t = x_t-1 + v_xy * dt
    kf.transitionMatrix = np.array([[1, 0, dt, 0],
                                    [0, 1, 0, dt],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]], dtype=np.float32)

    kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                     [0, 1, 0, 0]], np.float32)

    kf.measurementNoiseCov = np.array([[1, 0],
                                       [0, 1]], dtype=np.float32)

    # initialize the first state vector
    kf.statePost = np.array([0, 0, 0, 0], dtype=np.float32) * 0.01

    return kf

