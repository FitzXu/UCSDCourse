# CSE203B

+ Note
  + Out of Town 3/13-21 2019 
  + Last class of the quarter is 3/12
+ Logistics
  + Homework 30%
    + Exercises (From book)
    + Assignment
  + Project 40%
    + Theory of applications
    + Survey of state of art
    + Outlines, references
    + Presentation
    + Report
  + Exams (30%)
    + Midterm (Feb 12th) - up to Chapter 5 
  + Textbook Convex Optimization Stephen Boyd

## Formulation

### Introduction

1. General Format

   $min f_o(x)$

   subject to

   $f_i(x)\leq b_i$, $i=1,2,3...,m$

   $x=(x_1,x_2,...,x_n)^T$

   $f_i(x): R^n\rightarrow R$.

A: Static System

B: Dynamic System **[Will not cover in this course]**

​	$f_i(x) = g_i(x)$ $\bar x= \frac{dx}{xt}$

+ Convex Problem Definition
  1. $f_o(x)$ is a convex function
  2. $f_i(x)\leq b_i, i=1,2...m$  => $\{x|f_i(x)\leq b_i,i=1,2,...m\}$ is a convex set

+ Definition
  + A function is convex: $f_i(\alpha x +\beta y)\leq f_i(x)+\beta f_i(y)$ $x,y\in R^n, \alpha+\beta=1, \alpha,\beta\geq0$
  + A straight line is also a convex function 

## Discsion

### Matrix and vector

+ solve $Ax = b$
+ 

## Convexity

+ Set

  + Implicit Expression ( view from equations)

    + Ex: $S_1 = \{x |Ax\leq b\}$
    + Example:
      + $x_1+2x_2+3x_3\leq4$
      + $2x_1-x_2\leq 3$
      + $x_2+x_3\leq5​$
      + $x_3\leq 10$
    + $S = S_1\cap S_2\cap S_3\cap S_4$
    + Therom: The intersection ofconvex sets is convex
    + For any $x\in S,y\in S$, If $\alpha x + \beta y \in S $ for any $\alpha$ and $\beta$ then S is convex 

  + Diff beween convex funciton and convex set?

  + Statement: All linear equations imply a convex set. [For non-linear equations, it should be linearized]

  + Explicit Expression (View from enumerations) **equivalent representation to implicit expressi on**

    $\{\theta_1u_1+\theta_2u_2+...+\theta_ku_k|\theta_1+\theta_2+...\theta_k =1 ,\theta_i\geq 0\}$

    $S = {\theta_1u_1+\theta_2u_2+...+\theta_4u_4|\sum\theta_i=1}$

+ Affine, Cone, Convex Hull and Polyhedron
  + Convex Sets
    + Given $u_1,u_2,...u_k \in R^n$ $\theta_i\in R$.
    + $f(u,\theta)=\sum_{i=1}^n\theta_iu_i=\theta_1u_1+\theta_2u_2+...+\theta_ku_k$
  + Degrees of freedom
  + all vec in hyperplane are othogonal to vector $a$
  + 