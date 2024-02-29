# Computer Vision Projects
Learning by doing. This repo represents my journey of learning CV and will contain a variety of small-scale projects, from classic to CNN.  


### 1. Object Tracking
    
- Optical flow - Parking lot tracking

![Surface Lot Tracking](media/demo/1_OF_parkinglot_tracking.gif)

- Optical flow - Vehicle tracking

![Vehicle Tracking](media/demo/1_OF_vehicle_tracking.gif)


[**Repo**](1-object-tracking/optical-flow) | [**README**](1-object-tracking/optical-flow/README.md) | [**Demo**](https://www.youtube.com/watch?v=uecvioD0xVw)

#### Acknowledgement 

    The car detection model was trained on the pretrained YOLOv8 model developed by Ultralytics.
    https://github.com/ultralytics/ultralytics 

    The segementation model, Segment-Anything, was developed by Meta AI.
    https://github.com/facebookresearch/segment-anything 

### 2. Feature Detection

- Feature matching

![Feature Matching](media/demo/2-FT-temple.jpg)

- Stitch and map drone images onto satellite imagery

<div style="display: flex; justify-content: space-around;">
    <div style="width: 30%; text-align: left; margin-right: 3px;">
        <img src="media/out/feature_matching/satellite.jpg" alt="Satellite" style="width: 100%; margin-bottom: -8px;" ><br>
        <span style="font-size: 12px;">Satellite</span>
    </div>
    <div style="width: 30%, text-align: left; margin-right: 3px;">
        <img src="media/out/feature_matching/transparent.jpg" alt="Stitched" style="width: 100%; margin-bottom: -8px;"><br>
        <span style="font-size: 12px;">Stitched</span>
    </div>
    <div style="width: 30%, text-align: left; ">
        <img src="media/out/feature_matching/filled.jpg" alt="Overlay" style="width: 100%; margin-bottom: -8px;"><br>
        <span style="font-size: 12px;">Overlay</span>
    </div>
</div>

[**Repo**](2-feature-detection) | [**README**](2-feature-detection/README.md) 




