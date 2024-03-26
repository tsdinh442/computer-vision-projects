# PARKING OCCUPANCY ANALYSIS

Monitor and analyze parking occupancy at scale using drones and AI

![gif](../../media/demo/0_parking.gif)

#### 1. Collect drone images
![Drones](../../media/demo/collage.png)


#### 2. Collect high-resolution satellite imagery
![Satellite](../../media/demo/satellite.png)


#### 3. Feature detection and descriptor matching
![Feature Matching](../../media/demo/0_feature_matching.jpg)


#### 4. Apply RANSAC to filter outliers and compute homography matrix
![Feature Matching](../../media/demo/0_ransac.jpg)


#### 5. Warp transform

![Warp](../../media/demo/0_warped.jpg)


#### 6. Overlay drone images onto satellite imagery 
<img src="../../media/demo/0_trans_.jpg" alt="satellite" style="width: 48%;"> <img src="../../media/demo/0_filled_.jpg" alt="stitched" style="width: 48%;"> 


#### 7. Perform car detection
![Detect](../../media/demo/0_detect.JPG)


#### 7. Apply binary mask image
![Mask](../../media/demo/stall_mask_2.png)


#### 8. Label occupancy status for each parking stall 

<img src="../../media/demo/0_8AM.jpg" alt="satellite" style="width: 32%;"> <img src="../../media/demo/0_10AM.jpg" alt="stitched" style="width: 32%;"> <img src="../../media/demo/0_12PM.jpg" alt="overlaid" style="width: 32%;">
<img src="../../media/demo/0_2PM.jpg" alt="satellite" style="width: 32%;"> <img src="../../media/demo/0_4PM.jpg" alt="stitched" style="width: 32%;"> 
