## About

In this project we compose a panorama from five different images that illustrate the same 3D scene, from slightly different viewpoints. The main objective of this project is to implement the RANSAC algorithm (Random Sample Consensus) in order to calculate the homography transformation between each pair of images, that transforms one image to the coordinate frame of the other. Finally, by making use of the estimated homographies, we stitch the images together, thus forming a panoramic image.

## Prerequisites 

Install OpenCV:
```
pip install opencv-python
```

Install Matplotlib:
```
pip install matplotlib
```

Install NumPy:
```
pip install numpy
```

## Run:
```
python ImageStitching.py
```

## Python Version
Python 3.7.11

## Author

Theodoros Kyriakou

## Results

Input Images        
<img src="Results/Original_Images.png">

Panorama        
<img src="Results/Image_Stitching.png">

My implementation of RANSAC Inliers        
<img src="Results/My_implementation_of_RANSAC_Inliers.png">

RANSAC Inliers        
<img src="Results/RANSAC_Inliers.png">

SIFT Feature Matching        
<img src="Results/SIFT_Feature_matching.png">

SIFT Keypoints        
<img src="Results/Image_SIFT_Keypoints.png">


