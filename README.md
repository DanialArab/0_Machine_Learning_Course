# Machine Learning Course -- Coursera

This repository contains my learnings (documented through personal and course notes) along with my solutions to the programming assignments (using OCTAVE) of the Machine Learning course taught by Prof. Andrew Ng. Please respect the Coursera Honor Code in terms of assignment submissions. As a gentle reminder the Honor Code of the Machine Learning course is also included. 

# Table of content

1. [Machine Learning Class Honor Code](#1)

2. [A quick introduction](#2)

3. [ML algorithms](#3)
    1. [Supervised learning algorithms](#4)
    2. [Unsupervised learning algorithms](#5)
4. [Note on the programming assignments](#6)
5. [Linear Regression](#7)
6. [Gradient Descent](#8)
    1. [GD for univariate linear regression](#9)
    2. [GD for multivariate linear regression](#10)
7. [Practical tips to speed up GD convergence](#11)
8. [Normal equation](#12)
9. [Logistic regression for classification problem](#13)







<a name="1"></a>
# Machine Learning Class Honor Code
We strongly encourage students to form study groups, and discuss the lecture videos (including in-video questions). We also encourage you to get together with friends to watch the videos together as a group. However, the answers that you submit for the review questions should be your own work. For the programming exercises, you are welcome to discuss them with other students, discuss specific algorithms, properties of algorithms, etc.; we ask only that you not look at any source code written by a different student, nor show your solution code to other students.

<a name="2"></a>
# A quick introduction
There is not a consensus on the definition of machine learning (ML). One of the accepted definition is that ML is a field of study that gives computers the ability to learn without explicitly being programmed (Arthur Samuel, 1959). 

<a name="3"></a>
# ML algorithms
There are four main types of ML algorithms:
* Supervised learning (SL) algorithms, where we teach computer (the most common types of ML problems) 
* Unsupervised learning (USL) algorithms, where we let the computer learn by itself 
* Reinforcement learning algorithms 
* Recommender systems

<a name="4"></a>
## Supervised learning algorithms
In SL algorithms we have a dataset with right answers (i.e., we have an idea of a relationship between the input and output). These algorithms are categorized as:
* Regression, wehre we try to map the inputs to some **CONTINEOUS / REAL-VALUED** outputs 
* Classification, wehre we try to map the inputs to  **DISCRETE-VALUED** outputs   

<a name="5"></a>
## Unsupervised learning algorithms
In USL algorithms, the dataset does not have any label and we just try to find a structure within the data with zero or little prior knowledge, like clustering algorithms.  

<a name="6"></a>
# Note on the programming assignments

In the following, first a brief discussion/explanation on the topic is presented to clarify the algorithm. The corresponding codes are presented separately.

<a name="7"></a>
# Linear Regression 

Linear regression problems can be either univariate (one variable) or multivariate problems. In either case, we define a hypothesis function, called y_hat or h(theta), to make a prediction. We can measure the accuracy of our prediction using a cost function. One form of the cost function is mean squared error function:

![1](https://user-images.githubusercontent.com/54812742/137403944-02c95480-f8f3-40ae-b626-042c43f1d3b0.PNG)

where m refers to the number of training examples in a dataset, thetas are model parameters, and n is number of features/independant variables (in the above formula which is for univariate linear regression n is equal to 1). We use different optimization algorithms such as gradient descent to minimize the cost function, which is the difference between our prediction and the right answers. 

<a name="8"></a>
# Gradient Descent 
Gradient descent (GD) is a general useful optimization algorithm not only used in linear regression but also in many ML problems. The definition of GD is as follows:

![2](https://user-images.githubusercontent.com/54812742/137404024-d79226fe-5779-4bf7-9f16-d6973318b739.PNG)

where alpha is learning rate, denoting the size of the step we are taking to reach the local optima, and the derivate term indicates the direction. 
Very important point is that the update should be performed **SIMULTANEOUSLY**, as shown in the following:

![Simultaneous update](https://user-images.githubusercontent.com/54812742/137401027-6680d2f8-534d-49b1-93d4-c8fff94ff7fe.PNG)

Some useful points:
* Batch GD is when an algorithm looks at all of the examples in the **entire** training set on every step.
* As we approach a local minimum, GD automatically takes smaller steps and so there is no need to decrease learning rate over time
* Too small learning rate may lead to a too slow GD
* Too large learning rate may make GD to overshoot the minimum. It may fail to converge or even diverge

<a name="9"></a>
## GD for univariate linear regression
In univariate linear regression, n is equal to 1 and we only have one independent variable (x) and so our model parameters are theta_0 (the intercept) and theta_1 (the slope). In this case our hypothesis function would be h_theta(x) = theta_0 + theta_1 * x. If we plug in the mean squared error function (equation 1) into equation 2, we can obtain the GD formula specifically for univariate linear regression:

![11](https://user-images.githubusercontent.com/54812742/137406475-600c0784-c016-4a69-b1c1-57e0a015a492.PNG)

<a name="10"></a>
## GD for multivariate linear regression
The followings are some useful notations:

![notation](https://user-images.githubusercontent.com/54812742/137408353-8c1dfa30-dc86-4fb4-b6aa-6731ebc29a3c.PNG)

In this case, our hypothesis function is as follows:

![4](https://user-images.githubusercontent.com/54812742/137408629-27b5adce-2b1e-496c-b09a-7bd103843fc7.PNG)

For the sake of notation, we assume x_0 = 1 to allow matrix multiplication of theta and X. Then the multivariate hypothesis function, equation 4, can be represented using the definition of matrix multiplication:  

![5](https://user-images.githubusercontent.com/54812742/137409200-8728e703-9658-4d84-b395-713ed67d6efb.PNG)

In this case, the GD would be:

![66](https://user-images.githubusercontent.com/54812742/137410324-ed93ed69-194b-4851-96f5-1558a44ae2e4.PNG)

which can be summarized as:

![77](https://user-images.githubusercontent.com/54812742/137410328-5fd4af85-1dc7-4294-9291-b6ff182a7dff.PNG)

<a name="11"></a>
# Practical tips to speed up GD convergence

* Feature scaling using mean normalization through replacing the features with the corresponding values obtained from:

![mean norm](https://user-images.githubusercontent.com/54812742/137410775-d85f1bf5-f71f-422b-952f-b9cec96a02c1.PNG)

* Making sure the GD works well through monitoring cost function vs. number of iterations (cost function should decrease in every single iteration), in this case we need to make sure that the size of learning rate is not too small or too big
* We need to combine multiple features into one, if possible

<a name="12"></a>
# Normal equation
Normal equation (NE) gives us much better way to **analytically** solve for the optimal values of theta in linear regression problem. However, in more sophisticated learning algorithms like classification algorithms such as logistic regression algorithms NE does not work and we need to resort to GD. However, in linear regression with not too many number of features (less than 10000) NE is an efficient alternative to GD. 

First we construct a feature matrix (also called design matrix) which includes all the features from the training dataset plus a column of one, corresponding to x_0, also we construct a vector containing all the outputs. X would be m by (n+1) matrix and y would be m dimensional vector, as shown below. 

![desogn matrix](https://user-images.githubusercontent.com/54812742/137521844-8e6289b0-2fed-45be-82b2-b149965d3bd9.PNG)

Optimal theta can be calculated through the following equation:

![88](https://user-images.githubusercontent.com/54812742/137523767-418ca7d3-3d29-4e25-ac1e-ca5139d4e61f.PNG)

The following table summarizes the advantage and disadvantages of GD vs. NE. 

![table](https://user-images.githubusercontent.com/54812742/137523902-55d5ca9d-5ece-489d-b732-fdbe82a46bc7.PNG)

If n is 10000 we start thinking of using GD, for the larger values we go for GD because normal equation could be computationally very expensive due to calculating the inverse. Another advantage of NE is that we do not need to do feature scaling.

<a name="13"></a>
# Logistic regression for classification problem
Classification can be either binary or multi class classification.
The property of the logistic regression algorithm is that outputs or the predictions of the algorithm are always between zero and one and do not become either bigger than one or less than zero. 
The name of the algorithm is not intuitive because it is named logistic regression but it is used in classification. 
Logistic function (or sigmoid function) is the hypothesis function of the logistic regression algorithm, which is as follows:

![1](https://user-images.githubusercontent.com/54812742/137561885-0475be42-c99b-4939-b21c-302ef9dd0837.PNG)

The sigmoid function maps any real number to the (0, 1) interval, making it useful for transforming an arbitrary-valued function into a function better suited for classification. h_theta(x) will give us the probability that our output is 1:

![2](https://user-images.githubusercontent.com/54812742/137561946-af6d5316-f1e4-4fee-835b-c3bf4511da46.PNG)

## Decision boundary

In order to get our discrete 0 or 1 classification, we can translate the output of the hypothesis function as follows:

![3](https://user-images.githubusercontent.com/54812742/137561973-3a6f7d3f-5f40-4051-b37e-39bdb5ed981f.PNG)

The way our logistic function g behaves is that when its input is greater than or equal to zero, its output is greater than or equal to 0.5:

![0](https://user-images.githubusercontent.com/54812742/137562031-14a39b9a-4fc5-4925-8f2d-527c94c5c2bc.PNG)

![6](https://user-images.githubusercontent.com/54812742/137562040-7deba3ef-931e-42d5-b62a-a74382c0a61b.PNG)

So if our input to g is theta_transpose*X then that means:

![7](https://user-images.githubusercontent.com/54812742/137562067-7ae769af-3c7a-4a53-83f4-acbf0058da79.PNG)

From these statements we can now say:

![8](https://user-images.githubusercontent.com/54812742/137562090-d336ccd7-1aea-4da1-9b66-495a576fca1e.PNG)

The decision boundary is the line that separates the area where y = 0 and where y = 1. It is created by our hypothesis function. The decision boundary is a property, not of the training set, but of the hypothesis under the parameters.

## Nonlinear decision boundary

The input to the sigmoid function g(z), i.e., theta_transpose*X, doesn't need to be linear, and could be a function that describes a circle or any shape to fit our data. 

## Logistic regression – Cost function/optimization objective

We cannot use the same cost function that we use for linear regression, i.e., the squared error function, because the Logistic Function will cause the output to be wavy, causing many local optima. In other words, it will not be a convex function. Instead, our cost function for logistic regression looks like:

![Capture2](https://user-images.githubusercontent.com/54812742/137562105-06a32bc6-1b2e-4ea8-a881-73b7e4076409.PNG)

![10](https://user-images.githubusercontent.com/54812742/137562118-b9f770f1-aca9-405f-85f4-9e82e48eb249.PNG)

If our correct answer 'y' is 0, then the cost function will be 0 if our hypothesis function also outputs 0. If our hypothesis approaches 1, then the cost function will approach infinity. Likewise, if our correct answer 'y' is 1, then the cost function will be 0 if our hypothesis function outputs 1. If our hypothesis approaches 0, then the cost function will approach infinity.

Writing the cost function of the logistic regression in this way guarantees that J(θ) is convex for logistic regression. That is this particular choice of cost function leads to a convex optimization problem. In this case, overall cost function J of theta will be convex and local optima free.
We can compress our cost function's two conditional cases into one case:

![11](https://user-images.githubusercontent.com/54812742/137562142-f83ef8a0-ed95-479c-86c8-3c5b6226e483.PNG)

A **vectorized implementation** is:

![12](https://user-images.githubusercontent.com/54812742/137562153-8a645e77-8c78-42f2-84bd-a518be43c4f5.PNG)

the general form of gradient descent is:

![13](https://user-images.githubusercontent.com/54812742/137562171-ea58dc3c-7b30-4ee9-8ce0-d72670dfe4e5.PNG)

After applying derivative terms:

![14](https://user-images.githubusercontent.com/54812742/137562190-876ee448-1b67-42ab-95ff-9ab21716c437.PNG)

which is identical to the algorithm used in linear regression. The main difference is in h_theta(x), which was theta_transpose(x) in linear regression but is equal to sigmoid function, having theta_transpose(x) for z, in logistic regression. So, they are not actually similar. We still have to **simultaneously** update all values in theta.

A **vectorized implementation** is:

![15](https://user-images.githubusercontent.com/54812742/137562207-1123cfa1-1f4e-48aa-9a84-53b88abdbe13.PNG)

The idea of feature scaling also can speed up applying GD for logistic regression.

# Advanced Optimization Algorithms 

* "Conjugate gradient"
* "BFGS"
* "L-BFGS" 

These are more sophisticated, faster ways to optimize θ that can be used instead of gradient descent. The advantage of these algorithms is that there is no need to manually pick a learning rate value, because they do have an inner loop which takes care of it. Also, these algorithms are often faster than GD.  Their disadvantage is that they are more complex. 
All of these advanced algorithms are already written in libraries. We only need to provide code for J(theta) and the derivative term of J(theta) with respect to theta_j, as follows: 

function [jval, gradient] = CostFunction (theta)

   jval = [… code to compute J(theta) …];
   
   gradient  = [ .. code to compute derivative of J (theta) …];
   
end

and then we can use octave's "fminunc()" optimization algorithm along with the "optimset()" function that creates an object containing the options we want to send to "fminunc()":

options = optimset('GradObj', 'on', 'MaxIter', 100);

initialTheta = zeros(2,1);

   [optTheta, functionVal, exitFlag] = fminunc(@costFunction, initialTheta, options);
   
That is we give to the function "fminunc()" our cost function, our initial vector of theta values, and the "options" object that we created beforehand.

fminunc = function minimization unconstrained. Constraints in optimization often refer to constraints on the parameters, for example, constraints that bound the possible values theta can take (e.g., theta <= 1). Logistic regression does not have such constraints since theta is allowed to take any real value.

# Multiclass Classification

## One-vs-all (or one-vs-rest) algorithm 

In classification of data when we have more than two categories, instead of y = {0,1} we will expand our definition so that y = {0,1...n}.
Since y = {0,1...n}, we divide our problem into n+1 (+1 because the index starts at 0) binary classification problems; in each one, we predict the probability that 'y' is a member of one of our classes.

We are basically choosing one class and then lumping all the others into a single second class. We do this repeatedly, applying binary logistic regression to each case, and then use the hypothesis that returned the highest value as our prediction.

![16](https://user-images.githubusercontent.com/54812742/137644883-686f658a-9600-4f74-bb3b-f82ad5e69dd0.PNG)

# Overfitting probem and its solution called regularization

Underfitting, or high bias, is when the form of our hypothesis function h maps poorly to the trend of the data. It is usually caused by a function that is too simple or uses too few features. At the other extreme, overfitting, or high variance, is caused by a hypothesis function that fits the available data but does not generalize well to predict new data. It is usually caused by a complicated function that creates a lot of unnecessary curves and angles unrelated to the data.
This terminology is applied to both linear and logistic regression. There are two main options to address the issue of overfitting:

1) Reduce the number of features (the draw back is we lose some information which may be useful):
* Manually select which features to keep.

* Use a model selection algorithm (studied later in the course), which automatically determines which features need to be kept and which ones need to be throw out 
2) Regularization

* Keep all the features, but reduce the magnitude/values of parameters theta_j
*
* Regularization works well when we have a lot of slightly usseful features that we don’t want to throw them out

![17](https://user-images.githubusercontent.com/54812742/137644922-b7bccc45-b524-4b85-bf61-cd150b5c2a04.PNG)

![18](https://user-images.githubusercontent.com/54812742/137644927-37679853-8813-4be7-8719-b4d196bb18e9.PNG)

## How to apply regularization and the idea of regularized cost function

## How to apply regularization and the idea of regularized cost function – Linear regression

We can apply regularization to both linear regression and logistic regression. First, let’s work on linear regression. For linear regression problem, we can regularize all of our theta parameters in a single summation as:

![20](https://user-images.githubusercontent.com/54812742/137644959-4ac58fdc-c9ac-4e56-8115-388ed39eac35.PNG)

We do not penalize theta_0 and that is why we have j starting from 1 in the second summation term.
Lambda is regularization parameter, which controls a trade off between two different goals: the first goal, captured by the first term in the regularized cost function, is that how well we can fit the training set and the second goal is keeping the parameters small and therefore keeping the hypothesis function relatively simple to prevent overfitting 
If we set lambda to a too larger value, algorithm results in underfitting (fails to even fit the training set) because in this case we penalize all the parameters very heavily, which is like we end of having a hypothesis of only equal to theta_0 leading to underfitting problem. So, a good choice of regularization parameter is required. The idea of how to automatically choose lambda will be discussed when discussing the model selection algorithm. 

# GD to be applied on regularized cost function of linear regression 

![21](https://user-images.githubusercontent.com/54812742/137644978-5e20c3de-c313-4132-a8ca-8a1099855772.PNG)

Which can be written as:

![22](https://user-images.githubusercontent.com/54812742/137644985-e784c4f5-cda7-4ed3-a0e7-3fd8f5866ba9.PNG)

The term (1 – alpha*lambda/m) in the above equation will always be less than 1. Intuitively you can see it as reducing the value of theta_j by some amount on every update. The second term is now exactly the same as it was before.

# Normal equation to be applied on regularized cost function of linear regression

The modified form of the equation would be:  

![23](https://user-images.githubusercontent.com/54812742/137644996-478dc2a4-94e0-465c-9c26-e2c3eaed8595.PNG)

L is a matrix with 0 at the top left and 1's down the diagonal, with 0's everywhere else. It should have dimension (n+1)×(n+1). Intuitively, this is the identity matrix (though we are not including x_0), multiplied with a single real number λ.
Recall that if m < n, then X_transpose * X is non-invertible. However, when we add the term λ⋅L, then X_transpose * X + λ⋅L becomes invertible.

## How to apply regularization and the idea of regularized cost function – Logistic regression

We can regularize logistic regression in a similar way that we regularize linear regression. As a result, we can avoid overfitting. Both GD and the advanced optimization techniques will be discussed:

GD:

Our cost function for logistic regression (non – regularized version) is as follows:

![24](https://user-images.githubusercontent.com/54812742/137645013-a48f8387-03ba-47e4-bc3c-68e017759ba8.PNG)

We can regularize this equation by adding a term to the end:

![25](https://user-images.githubusercontent.com/54812742/137645036-5cb0a169-0b0a-49ec-8267-58e01c0d06a8.PNG)

GD:

![26](https://user-images.githubusercontent.com/54812742/137645047-6b9aafae-e18e-43c4-9d94-b4a341da0c04.PNG)

Again this equation seems identical to the one for linear regression, but the difference is in h_theta(x) which is equal to sigmoid function for logistic regression. 

# For the advanced optimization:

It is the same as before, I just need to modify Jval and gradient terms to include the regularization term:

![27](https://user-images.githubusercontent.com/54812742/137645066-0467e123-52f5-494e-bc47-c4ee8f4b5e74.PNG)







