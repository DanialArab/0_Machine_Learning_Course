# Machine Learning Course -- Taught by Prof. Andrew Ng

This repository contains my learnings along with my solutions to the programming assignments (using OCTAVE) of the Machine Learning class. Please respect the Coursera Honor Code in terms of assignment submissions. As a gentle reminder the Honor Code of the Machine Learning course is as follows: 

# Machine Learning Honor Code
We strongly encourage students to form study groups, and discuss the lecture videos (including in-video questions). We also encourage you to get together with friends to watch the videos together as a group. However, the answers that you submit for the review questions should be your own work. For the programming exercises, you are welcome to discuss them with other students, discuss specific algorithms, properties of algorithms, etc.; we ask only that you not look at any source code written by a different student, nor show your solution code to other students.

# A quick introduction:
There is not a consensus on the definition of machine learning (ML). One of the accepted definition is that ML is a field of study that gives computers the ability to learn without explicitly being programmed (Arthur Samuel, 1959). 

# ML algorithms:
There are three main types of ML algorithms:
* Supervised learning (SL) algorithms, where we teach computer (the most common types of ML problems) 
* Unsupervised learning (USL) algorithms, where we let the computer learn by itself 
* Reinforcement learning algorithms 
* Recommender systems

# Supervised learning algorithms:
In SL algorithms we have a dataset with right answers (i.e., we have an idea of a relationship between the input and output). These algorithms are categorized as:
* Regression, wehre we try to map the inputs to some **CONTINEOUS / REAL-VALUED** outputs 
* Classification, wehre we try to map the inputs to  **DISCRETE-VALUED** outputs   
 
 # Unsupervised learning algorithms:
In USL algorithms, the dataset does not have any label and we just try to find a structure within the data with zero or little prior knowledge, like Clustering algorithms.  

# Note on the assignments: 

In the following, a brief explanation for each problem has been presented:

# 1- Linear Regression 

Linear regression problems can be either univariate (one variable) or multivariate problems. In either case, we define a hypothesis function, called y_hat or h(theta), to make a prediction. We can measure the accuracy of our prediction using a cost function. One form of the cost function is mean squared error function:

![1](https://user-images.githubusercontent.com/54812742/137403944-02c95480-f8f3-40ae-b626-042c43f1d3b0.PNG)

where m refers to the number of training examples in a dataset, thetas are model parameters, and n is number of features/independant variables (in the above formula which is for univariate linear regression n is equal to 1). We use different optimization algorithms such as gradient descent to minimize the cost function, which is the difference between our prediction and the right answers. 

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

# GD for Univariate linear regression
In univariate linear regression, n is equal to 1 and we only have one independent variable (x) and so our model parameters are theta_0 (the intercept) and theta_1 (the slope). In this case our hypothesis function would be h_theta(x) = theta_0 + theta_1 * x. If we plug in the mean squared error function (equation 1) into equation 2, we can obtain the GD formula specifically for univariate linear regression:

![11](https://user-images.githubusercontent.com/54812742/137406475-600c0784-c016-4a69-b1c1-57e0a015a492.PNG)

# GD for Multivariate linear regression
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

# Practical tips to speed up GD convergence

* Feature scaling using mean normalization through replacing the features with the corresponding values obtained from:

![mean norm](https://user-images.githubusercontent.com/54812742/137410775-d85f1bf5-f71f-422b-952f-b9cec96a02c1.PNG)

* Making sure the GD works well through monitoring cost function vs. number of iterations (cost function should decrease in every single iteration), in this case we need to make sure that the size of learning rate is not too small or too big
* We need to combine multiple features into one, if possible
