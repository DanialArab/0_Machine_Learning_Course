# Machine-Learning-Course----Taught-by-Andrew-Ng

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
* Regression, wehre we try to map the inputs to some **CONTINUOUS / REAL-VALUED** outputs 
* Classification, wehre we try to map the inputs to  **DISCRETE-VALUED** outputs   
 
 # Unsupervised learning algorithms:
In USL algorithms, the dataset does not have any label and we just try to find a structure within the data with zero or little prior knowledge, like Clustering algorithms.  

# Note on the assignments: 

In the following, a brief explanation for each problem has been presented:

# 1- Linear Regression 

Linear regression problems can be either univariate (one variable) or multivariate problems. In either case, we define a hypothesis function, called y_hat or h(theta), to make a prediction. We can measure the accuracy of our prediction using a cost function. One form of the cost function is squared error function:

![cost_func_mean_sq_error](https://user-images.githubusercontent.com/54812742/137399110-7c3462a8-76c8-4065-8d14-d806f8d00a1d.PNG)

where m refers to the number of training examples in a dataset, thetas are model parameters, and n is number of features/independant variables (in the above formula which is for univariate linear regression n is equal to 1). We use different optimization algorithms such as gradient descent to minimize the cost function, which is the difference between our prediction and the right answers. 

# Gradient Descent 
Gradient descent (GD) is a general useful optimization algorithm not on ly used in linear regression but also in many ML problems. The definition of GD is as follows:

![GD definition](https://user-images.githubusercontent.com/54812742/137400383-9814feab-5627-43a2-9133-eea9b22a5019.PNG)

where alpha is learning rate, denoting the size of the step we are taking to reach the local optima, and the derivate term indicates the direction. 
Very important point is that the update should be performed **SIMULTANEOUSLY**, as shown in the following:

![Simultaneous update](https://user-images.githubusercontent.com/54812742/137401027-6680d2f8-534d-49b1-93d4-c8fff94ff7fe.PNG)




