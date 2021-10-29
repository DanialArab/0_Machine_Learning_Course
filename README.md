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
    1. [Decision boundary](#14)
    2. [Nonlinear decision boundary](#15) 
    3. [Logistic regression – cost function/optimization objective](#16)
    4. [Advanced optimization algorithms ](#17)
    5. [Multiclass classification -- One-vs-all (or one-vs-rest) algorithm ](#18)
10. [Overfitting problem and its solution (regularization)](#19)
    1. [Reducing the number of features](#20)
    2. [Regularization](#21)
          1. [How to apply regularization and the idea of regularized cost function – Linear regression](#22) 
              1. [GD to be applied on regularized cost function of linear regression](#23) 
              2. [Normal equation to be applied on regularized cost function of linear regression](#24) 
          2. [How to apply regularization and the idea of regularized cost function – Logistic regression](#25) 
              1. [GD to be applied on regularized cost function of logistic regression](#26)
              2. [Advanced optimization algorithms](#27)
            
10. [Neural Networks](#28)
    1. [Why Neural Networks](#29)
    2. [Model representation](#30)
    3. [Computing a Neural Network's Output](#31) 
    4. [Neural network cost function](#32) 
    5. [Backpropagation Algorithm to minimize neural network cost function](#33)
    6. [Numerical gradient checking](#34)  
    7. [Random Initialization](#35)  
    
              


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
The name of the algorithm is not intuitive because it is named logistic regression but it is used for classification. 
Logistic function (or sigmoid function) is the hypothesis function of the logistic regression algorithm, which is as follows:

![1](https://user-images.githubusercontent.com/54812742/137561885-0475be42-c99b-4939-b21c-302ef9dd0837.PNG)

The sigmoid function maps any real number to the (0, 1) interval, making it useful for transforming an arbitrary-valued function into a function better suited for classification. h_theta(x) will give us the probability that our output is 1:

![2](https://user-images.githubusercontent.com/54812742/137561946-af6d5316-f1e4-4fee-835b-c3bf4511da46.PNG)

<a name="14"></a>
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

<a name="15"></a>
### Nonlinear decision boundary

The input to the sigmoid function g(z), i.e., theta_transpose*X, doesn't need to be linear, and could be a function that describes a circle or any shape to fit our data. 

<a name="16"></a>
## Logistic regression – cost function/optimization objective

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

<a name="17"></a>
# Advanced optimization algorithms 

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

<a name="18"></a>
## Multiclass classification -- One-vs-all (or one-vs-rest) algorithm 

In classification of data when we have more than two categories, instead of y = {0,1} we will expand our definition so that y = {0,1...n}.
Since y = {0,1...n}, we divide our problem into n+1 (+1 because the index starts at 0) binary classification problems; in each one, we predict the probability that 'y' is a member of one of our classes.

We are basically choosing one class and then lumping all the others into a single second class (depicted in the below figure). 

![28](https://user-images.githubusercontent.com/54812742/137807543-ea6cb2c5-7f31-4e36-80ad-3051193a4a41.PNG)

We do this repeatedly, applying binary logistic regression to each case, and then use the hypothesis that returned the highest value as our prediction.

![16](https://user-images.githubusercontent.com/54812742/137644883-686f658a-9600-4f74-bb3b-f82ad5e69dd0.PNG)

<a name="19"></a>
# Overfitting problem and its solution (regularization)

Underfitting, or high bias, is when the form of our hypothesis function h maps poorly to the trend of the data. It is usually caused by a function that is too simple or uses too few features. At the other extreme, overfitting, or high variance, is caused by a hypothesis function that fits the available data but does not generalize well to predict new data. It is usually caused by a complicated function that creates a lot of unnecessary curves and angles unrelated to the data.
This terminology is applied to both linear and logistic regression. There are two main options to address the issue of overfitting:

<a name="20"></a>
## 1) Reducing the number of features (the draw back is we lose some information which may be useful):

* Manually select which features to keep.

* Use a model selection algorithm (studied later in the course), which automatically determines which features need to be kept and which ones need to be thrown out 

<a name="21"></a>
## 2) Regularization

* Keep all the features, but reduce the magnitude/values of parameters theta_j

* Regularization works well when we have a lot of slightly useful features that we don’t want to throw them out

![17](https://user-images.githubusercontent.com/54812742/137644922-b7bccc45-b524-4b85-bf61-cd150b5c2a04.PNG)

![18](https://user-images.githubusercontent.com/54812742/137644927-37679853-8813-4be7-8719-b4d196bb18e9.PNG)

<a name="22"></a>
### How to apply regularization and the idea of regularized cost function – Linear regression

We can apply regularization to both linear regression and logistic regression. First, let’s work on linear regression. For linear regression problem, we can regularize all of our theta parameters in a single summation as:

![20](https://user-images.githubusercontent.com/54812742/137644959-4ac58fdc-c9ac-4e56-8115-388ed39eac35.PNG)

We do not penalize theta_0 and that is why we have j starting from 1 in the second summation term.
Lambda is regularization parameter, which controls a trade off between two different goals: the first goal, captured by the first term in the regularized cost function, is that how well we can fit the training set and the second goal is keeping the parameters small and therefore keeping the hypothesis function relatively simple to prevent overfitting 

If we set lambda to a too larger value, algorithm results in underfitting (fails to even fit the training set) because in this case we penalize all the parameters very heavily, which is like we end of having a hypothesis of only equal to theta_0 leading to underfitting problem. So, a good choice of regularization parameter is required. The idea of how to automatically choose lambda will be discussed when discussing the model selection algorithm. 

<a name="23"></a>
#### GD to be applied on regularized cost function of linear regression 

![21](https://user-images.githubusercontent.com/54812742/137644978-5e20c3de-c313-4132-a8ca-8a1099855772.PNG)

Which can be written as:

![22](https://user-images.githubusercontent.com/54812742/137644985-e784c4f5-cda7-4ed3-a0e7-3fd8f5866ba9.PNG)

The term (1 – alpha*lambda/m) in the above equation will always be less than 1. Intuitively you can see it as reducing the value of theta_j by some amount on every update. The second term is now exactly the same as it was before.

<a name="24"></a>
#### Normal equation to be applied on regularized cost function of linear regression

The modified form of the equation would be:  

![23](https://user-images.githubusercontent.com/54812742/137644996-478dc2a4-94e0-465c-9c26-e2c3eaed8595.PNG)

L is a matrix with 0 at the top left and 1's down the diagonal, with 0's everywhere else. It should have dimension (n+1)×(n+1). Intuitively, this is the identity matrix (though we are not including x_0), multiplied with a single real number λ.
Recall that if m < n, then X_transpose * X is non-invertible. However, when we add the term λ⋅L, then X_transpose * X + λ⋅L becomes invertible.

<a name="25"></a>
### How to apply regularization and the idea of regularized cost function – Logistic regression

We can regularize logistic regression in a similar way that we regularize linear regression. As a result, we can avoid overfitting. Both GD and the advanced optimization techniques will be discussed:

<a name="26"></a>
#### GD to be applied on regularized cost function of logistic regression

Our cost function for logistic regression (non – regularized version) is as follows:

![24](https://user-images.githubusercontent.com/54812742/137645013-a48f8387-03ba-47e4-bc3c-68e017759ba8.PNG)

We can regularize this equation by adding a term to the end:

![25](https://user-images.githubusercontent.com/54812742/137645036-5cb0a169-0b0a-49ec-8267-58e01c0d06a8.PNG)

and: 

![26](https://user-images.githubusercontent.com/54812742/137645047-6b9aafae-e18e-43c4-9d94-b4a341da0c04.PNG)

Again this equation seems identical to the one for linear regression, but the difference is in h_theta(x) which is equal to sigmoid function for logistic regression. 

<a name="27"></a>
#### Advanced optimization algorithms 

It is the same as before, it is just needed to modify Jval and gradient terms to include the regularization term:

![27](https://user-images.githubusercontent.com/54812742/137645066-0467e123-52f5-494e-bc47-c4ee8f4b5e74.PNG)

<a name="28"></a>
# Neural Networks

<a name="29"></a>
## Why neural networks: 

Since we have logistic regression and linear regression, why do we need neural networks?
In a supervised learning classification problem with the following training dataset, we can apply logistic regression with a lot of nonlinear features. In this case, through having enough polynomial terms we can end up with a hypothesis that can separate positive and negative examples. 

![1](https://user-images.githubusercontent.com/54812742/139180480-b630dd67-76a7-4787-8370-92842ee55789.PNG)

This technique works well if we only have two features x_1 and x_2. What about we had much more features? Let’s say we have 100 features, and we want to include all the quadratic terms that is second order or polynomial terms, doing so we end up having a lot of features, around 5000, which is very expensive (roughly around n^2/2, where n is the number of features we originally had). We may include only a subset of these features, in this case, the subset features may not be enough and we cannot fit the data like the magenta line in the above figure. Therefore, simple logistic regression algorithm with adding more quadratic or cubic features is not a good idea to learn complex nonlinear hypothesis when the original number of features is large. In these cases, neural networks come to play a great role. 

<a name="30"></a>
## Model representation

How we represent our hypothesis when using neural networks? 

A neuron/node is a computational unit that gets a number of inputs and does some computation and outputs to other nodes in the network (below figure).

![2](https://user-images.githubusercontent.com/54812742/139182544-5396f2f2-0a97-46dc-8454-2377d36b9b6b.PNG)

This simple diagram represents a single neuron, while a neural network is just a group of these different neurons strong together (below figure). 

![neural network representation from NN class](https://user-images.githubusercontent.com/54812742/139187209-18344c7a-33fb-4475-9a17-2108c63538a3.PNG)

Usually, we do not count the input layer and so we call the above network a 2 layer neural network.

<a name="31"></a>
## Computing a Neural Network's Output

The logistic regression really represents two steps of computation. First z is computed and then, the activation is computed as a sigmoid function of z (below figure). The activation is the value that's computed by and as output by a specific neuron. So, a neural network just does this a lot more times.

![nn](https://user-images.githubusercontent.com/54812742/139187269-f7c284d4-c30b-4802-be6f-158d4d436d78.PNG)

<a name="32"></a>
# Neural network cost function 

Neural networks are one of the most powerful learning algorithms. The cost function for a neural network, for a classification application, is a generalized version of the logistic regression cost function:

![2](https://user-images.githubusercontent.com/54812742/139336809-b330dc57-e129-4664-b79d-4921dbfaf114.PNG)

where L , s_l, and K are the total number of layers in the network, number of units in layer l (not counting bias unit), and the number of output units/classes, respectively.

Some notes:

* the double sum simply adds up the logistic regression costs calculated for each cell in the output layer
* the triple sum simply adds up the squares of all the individual Θs in the entire network.
* the i in the triple sum does **not** refer to training example i

<a name="33"></a>
# Backpropagation Algorithm to minimize neural network cost function

"Backpropagation" is neural-network terminology for minimizing our cost function, just like what we were doing with gradient descent in logistic and linear regression. That is, we want to minimize our cost function J using an optimal set of parameters in theta. 

<a name="34"></a>
# Numerical gradient checking

There is a chance that although the neural network is working and the cost function is decreasing in every iteration, there is the high-level errors associated with the neural network. To check this, we need to do the numerical gradient checking:
The approximate to the derivative of our cost function would be:

![4](https://user-images.githubusercontent.com/54812742/139351691-a661d815-399a-4d26-9172-f5b8bde4637a.PNG)

With multiple theta matrices, we can approximate the derivative with respect to theta_j as follows:

![5](https://user-images.githubusercontent.com/54812742/139351714-f5fa428c-397f-4844-a7b7-43d71e46cd5a.PNG)

A small value for epsilon such as 0.0001, guarantees that the math works out properly. If the value for epsilon is too small, we can end up with numerical problems. We are only adding or subtracting epsilon to the **theta_j**.

![3](https://user-images.githubusercontent.com/54812742/139351741-c27657d1-aab3-4e44-bd9f-07538667043b.PNG)

<a name="35"></a>
# Random Initialization 

Whereas the zero-initialization worked okay when we were using logistic regression, initializing all of the parameters to zero actually **does not** work when training a neural network. Initializing all theta weights to zero does not work with neural networks because in backpropagate all nodes will update to the same value repeatedly (below figure). Instead, we can randomly initialize our weights for our theta matrices.

![6](https://user-images.githubusercontent.com/54812742/139351792-8e3af0ff-911b-446d-bfe0-bdedce93b422.PNG)



