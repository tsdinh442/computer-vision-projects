import sys
sys.path.append('../optical_flow')
import numpy as np
import cv2
from optical_flow import lucas_kanade
from utils import select_points, mark_dots, masking, out_of_bound, points, manhattan_distance, center_text, COLOR
from kalman import kalman_filter, Kalman_Filter
from yolov8 import detect

global COLOR

def track(video_path):
    """

    :param video_path:
    :return:
    """

    threshold = 20

    # stream and resize video frames
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    #frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

    # define an empty list to store all kalman filters
    kalman_filters = []
    n = 0
    while ret:

        if not ret or (cv2.waitKey(33) & 0xFF == ord('q')):
            break

        frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        #frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        _, centroids, bboxes, _ = detect(targeted_regions=[], image=frame, conf=0.85, iou=0.7)

        #for bbox in bboxes:
        #    p1 = int(bbox[:2][0]), int(bbox[:2][1])
        #    p2 = int(bbox[2:][0]), int(bbox[2:][1])
        #    cv2.rectangle(frame, p1, p2, COLOR, thickness=2)

        corrected_pos = list(map(lambda f: f.Estimate()[:2], kalman_filters))

        if len(centroids) > 0:
            centroids = np.array(centroids)
            centroids_reshape = centroids[:, np.newaxis, :]
            #frame, _ = mark_dots(frame, corrected_pos, radius=10, color=(255, 0, 0), thickness=1)

            if len(corrected_pos) > 0:
                distances = np.linalg.norm(np.array(corrected_pos) - centroids_reshape, axis=2)
                print(distances)
                tracked_centroid_indices = []

                for centroid_idx, distance in enumerate(distances):
                    min_distance = np.min(distance)
                    filter_idx,  = np.where(distance == min_distance)[0]

                    if min_distance < threshold:
                        tracked_centroid_indices.append(centroid_idx)
                        measurement = centroids[centroid_idx].astype(np.float32)
                        kalman_filters[filter_idx].Correct(measurement)
                        kalman_filters[filter_idx].time = 0
                        kalman_filters[filter_idx].Bbox(bboxes[centroid_idx])

                mask = np.ones(centroids.shape[0], dtype=bool)
                if tracked_centroid_indices:
                    mask[np.array(tracked_centroid_indices)] = False

                untracked_centroids = centroids[mask]

            else:
                untracked_centroids = centroids

            for u_centroid in untracked_centroids:
                kf = Kalman_Filter(*u_centroid)
                kalman_filters.append(kf)

        for idx, f in enumerate(kalman_filters):
            f.Check()
            print(f.tracked_point)
            predicted_pos = f.Predict()
            if f.bbox:
                if predicted_pos is not None:
                    predicted_pos = predicted_pos[:2].transpose().astype(np.int32)[0]
                    cv2.circle(frame, predicted_pos, radius=20, color=COLOR, thickness=-1)

                    #cv2.rectangle(frame, f.bbox[0], f.bbox[1], COLOR, thickness=2)
                    frame = center_text(frame, predicted_pos, str(idx))

            f.Refresh()

        #cv2.rectangle(frame, (0, 0), (frame.shape[1] - 1, 100), (0, 0, 0), thickness=cv2.FILLED)
        #cv2.putText(frame, "Kalman Filter", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR, 1, cv2.LINE_AA)
        cv2.imshow("output", frame)
        cv2.imwrite(f"../../media/out/tracked/{n}.jpg", frame)
        ret, frame = cap.read()
        n += 1

    cap.release()
    return

if __name__ == '__main__':

    # media path
    path = "../../media/videos/7.mp4"

    track(path)

    # destroy and exit
    cv2.destroyAllWindows()
