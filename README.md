# Computer Vision Projects
Learning by doing. This repo represents my journey of learning CV and will contain a variety of small-scale projects, from classic to CNN.  

### 0. Capstone project

- Monitor and analyze parking occupancy at scale using drones and AI

![gif](media/demo/0_parking.gif)

Techniques: 

    - Feature detection and descriptor matching
    - RANSAC
    - Homography matrix 
    - GIS
    - Object detection
    - Perspective transform
    
[**README**](0_capstone/parking_occupancy_monitoring/README.md) 

### 1. Object Tracking

- Kalman Filter - Multi-object tracking and tag them with an ID

<a href="https://www.youtube.com/watch?v=lNpc3wB2L2E">
  <img src="https://img.youtube.com/vi/lNpc3wB2L2E/0.jpg" width="100%" alt="YOUTUBE">
</a>

- Optical flow - Parking lot tracking

![Surface Lot Tracking](media/demo/1_OF_parkinglot_tracking.gif)

- Optical flow - Vehicle tracking

![Vehicle Tracking](media/demo/1_OF_vehicle_tracking.gif)


[**Repo**](1_object_tracking/optical_flow) | [**README**](1_object_tracking/optical_flow/README.md) | [**Demo**](https://www.youtube.com/watch?v=uecvioD0xVw)

#### Acknowledgement 

    The car detection model was trained on the pretrained YOLOv8 model developed by Ultralytics.
    https://github.com/ultralytics/ultralytics 

    The segementation model, Segment-Anything, was developed by Meta AI.
    https://github.com/facebookresearch/segment-anything 

### 2. Feature Detection

- Feature matching

![Feature Matching](media/demo/2-FT-temple.jpg)

- Feature matching + Homography - Stitch and map drone images onto satellite imagery

<img src="media/demo/2-FT-satellite.jpg" alt="satellite" style="width: 32%;"> <img src="media/demo/2-FT-stitched.jpg" alt="stitched" style="width: 32%;"> <img src="media/demo/2-FT-overlay.jpg" alt="overlaid" style="width: 32%;">

[**Repo**](2_feature_detection) | [**README**](2_feature_detection/README.md) 


### 3. Stereo Vision

- Epipolar lines

![epipolar](media/demo/3_ST_epilines.png)

- Depth map
![depth_map](media/demo/3_ST_depth_map.png)

[**Repo**](3_stereo_vision) | 





