## What Can Help Pedestrian Detection?

### Main idea

Aggregating extra features. Proposed **HyperLearner**.

### Introduction

+ challenge
  + less discriminable from the background
    + pedestrian always low resolution
    + misclassified with hard example, traffic sign, pillar boxes
  + accurate localization
    + occlussion by the crowd
    + one solution: extra low-level apparent features
+ Contributions
  +  integrate extra features
  + three groups of channel features
    + apparent-to-semantic channels
    + temporal channels
    + depth channels
  + HyperLearner: In HyperLearner, channel features are aggregated as supervision instead of extra
    inputs, and hence it is able to utilize the information of given features and improve detection performance while requiring no extra inputs in inference???

### Channel features for pedestrian detection

+ KITTI dataset
  + 7, 481 labeled images, resolution 1250 x 375 (training)
  + 7518 images for testing
  + Evaluate under by **PASCAL criteria** and three levels
+ Faster RCNN
  + anchors: 5 scales and 7 ratios
  + conv5 removed
+ Introduction to channel features
  + Apparent-to-semantic channels
    + ICF channel
    + edge extracted by HED network
  + Temporal channels
    + optical flow channel
  + Depth channels
    + turn to DispNet [21] to reconstruct the disparity channel

+ Integration technique
  + side branch to accept feature channel
  + result: utilize feature channel is better than the original baseline Faster RCNN
  + setting: two convolutional layers with **Guassian intialization**

