# CSE253

Fist Assignment Due Jan 17th

Each assignment says in one para about what contribute for each member.

Reading.

## Polynomial Curve Fit 

+ Loss Function - Sum of Square Error
  + $E(w) = \frac{1}{2}\sum\{{y(x_n,w)-t_n}\}^2$
  + Minimize the distance
+ How much degree of the order ?
  + 3th perfect
  + 9th overfit!!! 
+ Root Mean Square Error
  + Enable the test error to trace train error
+ Method to alleviate the overfit issue
  + Get more data. E.g. Flip the data, Rotate, Crop
  + Change the Loss Function to $J = E+\lambda C$ **(Regularization)**
  + $C = \frac{w^2}{w^2+1}$
  + Standard Way: (1) $C = |w|$ (2) $C = ||\vec \omega||_2$
  + $E(w) = \frac{1}{2}\sum\{{y(x_n,w)-t_n}\}^2+\frac{\lambda}{2}||\vec\omega||_2$
  + Cross-validation to set meta parameter($\lambda,m(order )$)
  + Early Stopping
    + use hold-out set to find the minimum loss
+ Face Recognition Example
  + PCA: extract only some portion of features
    + Subtract mean
    + $min(d-1, n-1)$ feature to extract
    + covariance matrix
+ Neural Networks 
  + Linear Regression
    + Supervised Learning
    + $\begin{align}\begin{bmatrix}x_1^1&x_2^1&x_d^1\\x_1^2&x_2^2&x_d^2\\x_1^N&x_N^1&x_d^N\end{bmatrix}\end{align}$
    + 







