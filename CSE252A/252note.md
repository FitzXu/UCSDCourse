# Lecture2

Reading: Chapters 1 & 2 of Forsyth & Ponce

## Human vision system

###Ways to study human vision

+ Cellular recording

  小猫对于line direction的感知，line的方向是stimulus，呈现结果是不同的line方向在不同receptive-field location上起作用

+ Functional MRI

  activation in the right fusiform gyrus

###Computational Modeling

物体的不同距离，Lens的形状将会发生改变。

<img src="https://i.loli.net/2018/12/13/5c1142278530e.png">

**The range of light:**

lighting range越来越小100000Lux到最后0.0001Lux

Direct sun $\rightarrow$sunny day$\rightarrow $cloudy day ... $\rightarrow $clear moonless night $\rightarrow $cloudy moonless night

电子照片electronic image的range在400Lux到50000Lux之间。

1 Lux = 1 $lumin/m^2$

**Rods and Cones**

Rod: 圆柱，$10^8$,在边缘

Cones: Conical 圆锥 $5\times10^6$ Fovea

三种types of cones: RGB, Shortwave(B), Midwave(G), Longwave(R).

**Rods and Cones 的分布**

相对于fovea，Cones的receptor在最中间，Rods在两侧，peak出现在10～20degree的位置，blind spot位置没有receptor。

<img src="https://i.loli.net/2018/12/13/5c1148442cffd.png">

##Other eyes

Trilobite Visual System 最古老的visual system

# Lecture3

Images are two-dimensional patterns of brightness values. 

## Pinhole Camera: Perspective Projection

Equation of Perspective Projection

给定空间中的一个点$P(x,y,z)$, 焦距是f，那么其投影是$P' = (f'x/z,f'y/z,f')$. 如果在image的中心建立一个image cordinates的话，那么P的投影点是
$$
P(x,y,z)\rightarrow P'(f'x/z,f'y/z)
$$

##Projective Geometry and Homogenous Coordinates 
### Homogenous Coordinates 

使用三个坐标来表示Projective Plane上第一个点，因为这样能够表示**无穷远处的点**。

Points at infinity – zero for last coordinate e.g., (x,y,0)。

**Conversion**

2D空间

Euclidean -> Homogenous: $(x,y)\rightarrow\lambda(x,y,1)$

The intensity of a pixel is
$$
I(x,y) = a(x,y)n(x,y)s
$$
a is the albedo, n is normal vector and s is the light source direction

Projective transformation 

**vanishing point **

is the perspective projection of that point at infinity, resulting from multiplication by the camera matrix.

**What is Camera**

# BRDF

# Segmentation

## 1. Edge

+ Edge come from?

Intensity dicontinous. white-> black  black -> white

+ Physical Edge come from?

object boundery

material property (pameter: a in Lambertian equation)

+ Noisy step edge

Cause the edge is where derivative is high, so in order to detected, smooth first (noise's derivative is also high).

+ Where to find egde for 1D edge.

First derivative is maximum.

Second derivative is zero.

+ how to compute in image.

Using Taylor series to expand around $x_0$
$$
\begin{align}f(x) = f(x_0)+f^{'}(x_0)(x-x_0)+\frac{1}{2}f^{''}(x_0)(x-x_0)^{2}+...\end{align}
$$
Substitute $x$ with $x_0+h$ and $x_0-h$
$$
f^{'}(x_0) = \frac{f(x_0+h)-f(x_0+h)}{2h}
$$

$$
f^{''}(x_0) = \frac{f(x_0+h)-2f(x_0)+f(x_0+h)}{2h}
$$

+ Convolve with kernel

1. First derivative [-1 0 1]
2. Second derivative [1 -2 1]

+ Canny Edge Detection

1. single threshold: easy to lose some edge
2. two different thresholds are ideal

## 2. Boundary

+ Difference from edge

  edge is the location where gradient diff is large. But boundary is the location to distinguish the object from the comtext.

+ Segmentation

  Divide each image into pieces, where each piece is a distinguished object

+ Precision and Recall
  1. Precision is the fractionof true positive rather than false positive
  2. Recall is the fraction of true positive rather than missed objects

## 3. Corner Detection

+ panorama stiching
  1. extract features
  2. match features
  3. combine

+ Basic Idea 

  Corner: change in all directions

+ Distribution of gradient

  + Flat: Near the region of (0,0)
  + Edge: Near one of the axis
  + Corner: Both dy and dx have different points

+ Finding Corner
  $$
  C(x,y) = \begin{bmatrix}I_x^2&I_xI_y\\I_xI_y&I_y^2\end{bmatrix}
  $$

  $$
  C = R^{-1}\begin{bmatrix}\lambda_1&0\\0&\lambda_2\end{bmatrix}R
  $$








+ Corner Detector (Parameters: Gaussian std window and threshold)
  + filter
  + compute the gradient
  + move window to construct C matrix
  + find $\lambda_1$and $\lambda_2$
  + if both values are large, that's a corner

+ Binocular Stereo

  Given two images of a scene, estimate the depth

  Core Problem: Correspondence two images

+ Correspondence

  + Triangulation: Straightfoward

+ Stereo Vision Outline

  + Offline: Calibrate camera

+ Reconstruction 3D

+ Epipolar Constraint  (Calibrated case)

  vector $\vec{OP}$,$\vec{O^{'}P^{'}}$ and $\vec{OO'}$ are coplanar

$$
\vec{OP} \cdot [\vec{OO'}\times\vec{O'P'}]=0
$$

$$
p\cdot [t\times (Rp')] = 0
$$

Method1: Using extrinsic and intrinsic matrix

Method2: Given 8 matching points to compute the essential matrix

+ Forward motion

  Base line  are the line between two ceners of the images

+ Uncalibrated case
  $$
  (H^{-1}q)^TE(H^{'-1}q^{'})=0\\
  q^T((H^{-1})^TE(H^{'-1}))q^{'T}=0\\
  q^TFq^{'T} = 0
  $$
  Can be solved using 8 points algorithm.

+ Compare window

  + SSD: sum square differences
  + dot product between two windows (cos)




# Motion structure Lec14 

Goal: two or more input image, and to estimate the camera motion and 3D structure of scene.

Total number of unknown $(M-1)\times 6 +3\times N +1$

Total number of measurements $2\times M\times N$$

Solution is Possible when $(M-1)\times 6 +3\times N +1 \leq 2\times M\times N$

+ RANSac
+ Motion Fileld

  + Focus of Expansion: intersection of velocity

+ Regid motion 
  + translation velocity $[\omega_x,\omega_y,\omega_z]$
  + angular volocity $[T_x,T_y,T_z]$
+ Motion Field Equation
  + Pure Translation
  + Pure Rotation
  + Estimate Depth

+ Small motition

+ Optical Flow 

  means "apparent motion of brightness patterns"
  + Constraint Equation

    Optical Flow: $(u,v)$ what we want to solve

    Assume brightness of patch remains in both images

    $I(x,y,t) = I(x+u\delta t,y+v\delta t,t+\delta t)$

    Apply Taylor series expansion

    $I(x,y,t) + \delta x \frac{\part I}{\part x}+\delta y \frac{\part I}{\part y}+\delta t \frac{\part I}{\part t}=I(x,y,t) $

    Substracting $I(x,y,t) $ and devided by $\delta t$.

    $\frac{\part I}{\part t}$ consider stacking 3 images (pixels) at (t-1,t,t+1) convolve with vector $[-1,0,1]$.

    **Goal: solve for $\frac{dx}{dt}$and $\frac{dy}{dt}$ but only one equation.**

    The Solution is one 

# Visual tracking

Main tracking notions:

- state
- dynamics
- represntation
- prediction
- data association
- correction
- initialization

## State

A clock t

state $s_{t-1},s_{t},s_{t+1}$

observation: 

tracker (3 main steps)

+ prediction

# Recognition 

Object recognition

- determine which object in the image  (2D)
- determine the pose of the object

Challenge

+ within-class variability
+ lighting
+ occlusion
+ clutter in background

Recognition progression

+ template matching
+ image- > feature vector <==> feature vector< - test image (Metric)
+ image -> depth map <==> 3D shape or other abstract representation <- test image
+ image -> feature vector <==> classifier <- feature vector <- test image  **(Classificaiton)**
+ images ->deep network <-test image

Pattern Recognition Architecture

​	Image -> feature extraction -> classification -> oject identity

Feature

Classifier

+ unsupervised

+ supervised
  + nearest neighbor classifier
    + temoplate matching 
    + dependent on distance function
  + bayesian classifier

Evaluation a binary classfier

+ $Precision = \frac{tp}{tp+fp} $
+ $Recall = \frac{tp}{tp+fn}$
+ ROC Curve

Introduction of sea bass/salmon classfication

+ conditional probability $P(x|w_1)$ and $P(x|w_2)$

+ In order to compute the likelihood of the class given the feature x

$$
P(w_j|x) = \frac{P(x|w_j)P(w_j)}{P(x)}
$$

+ And classification: $j = arg\;max\;g_j(x) $

+ How to get the probability of $P(x|w_j)$

  + assume that the class conditional is a kind of known distribution
  + kth nearest neighbor classification

+ Support vector machine

  + Goal is to gind a hyperplane that divides S into two clases
  + maximizing the distance of the plane to the near points of the two classes

+ Dimension Reduction: linear projection
  $$
  y = w^Tx
  $$

  + How to choose a good W? 
  + Principal Component Analysis (PCA) Eigenface

+ PCA for recognition

  + compute the mean imgae 
  + compute k eigenfaces