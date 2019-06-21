# Cse250b 

+ Landscape
  + Supervised learning
  + Unsupervised learning - finding good representations
  + Learning through interaction
+ Prediction problem can be categorized by the output space[discrete, continuous and probability values]
+ Discrete Output space: classification
  + Binary classification
  + Multi - class
  + Sructured Output e.g. Parsing, input is a sentence and output is a **parse tree**.
+ Continuous Output Space: Regression
+ Probabilities
  + E.g. credit card transaction. Input: detail of a transaction Output: the probability of being fraudulent.
  + Why not binary classficaiton? Sometimes need the probability to make decision.
+ Unsupervised Learning - Find Structure in data

##Jan 11

+ Nearest Neighbour Classfier
  + Find the nearest one 
  + How to measure the closeness? 
    + Stretch the image into a vector
    + Distance Function
      + Euclidean Distance
  + A way to measure the quality of Classfier: Test Error
    + A random classifier can achieve 90% test error
    + NN can achieve 3.09%
  + Ideas to improve
    + k-NN: find k closest images
    + new distance functions
      + Invariant: translation and roation
      + Shape context
  + Methods to measure the performace
    + Hold-out Set
    + Leave one out cross validation
      + Pros: BigTraining set and Testset
      + Cons: Expensive Computation
    + Compromise Version: 10-fold cross validation

## Jan 14

+ Feature Selection
  + Find the best weight to reweight the distance funciton
  + $\sum_i{w_i(x_i-x_i^{'})^2}$

+ Algorithm Issue: speeding up NN Search
  + K-d tree
    + Choose one corodinate (compute the variance: pick the maximum spread)
    + compute the median
    + split the space by one coordinate
    + narrow the search space half by half
  + Issue: the query near the border
    + draw the bowl around the query point
    + if the bowl overlaps one border, it means we should search the neighbour box
  + Issue: 
    + storage to store the data $O(nd)$
    + Time to compute  distance $O(d)$
    + Geometry
      + possible $2^{O(d)}$ points that have the same equivalent distance
+ Families about Distance Functions
  + $l_p$ functions $\|x-z\|_p = (\sum|x_i-z_i|^p)^{\frac{1}{p}}$
    + $l_2$
    + $l_\infty$ $max(x_i-z_i)$
    + $l_1 \sum_(x_i-z_i)$ 
  + Metric spaces
    + Non-negative
    + $d(x,y)=0 $ only when $x=y$
    + $d(x,y)=d(y,x)$
    + Kullback-Leibler distance

## Jan 16

Data Distribution P $(x,y) ~ P$











































+ 

+ 

+ 

+ 

+ 

+ 

+ q


