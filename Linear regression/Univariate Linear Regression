% Univariate Linear Regression 
clear;
close all;
clc;

% Step_1: Loading the data:
data = load('ex1data1.txt'); 

% Step_2: Plotting the data
fprintf('Plotting the data...\n');

X = data(:, 1); 
y = data(:,2);
m = size(X,1); # number of training examples 

plot(X,y, 'rx', 'markersize', 7)

% Step_3: Add a column of ones to X for the sake of notation to allow matrix multiplication between theta and X
num_parameters = size(X,2) + 1;
theta = zeros(num_parameters, 1);
X = [ones(m,1), X];

% Step_4: GD settings:
alpha = 0.01;
iterations = 1500;

% Step_5: Computing the cost function:

function J = Cmpute_Cost_Univariate (theta, X, y)
  m = size(y,1);
  J = 1 / ( 2 * m ) * sum (((X * theta) - y).^2);
end

J = Cmpute_Cost_Univariate (theta, X, y);

% Step_6: Applying GD to minimize cost function
function [theta, J_history] = GD_Univariate (X, y, theta, alpha, iterations)
  J_history = zeros (iterations, 1);
  m = size(y,1);
  for i = 1: iterations
  term_1 = (alpha / m) * sum (((X * theta) - y).* X(:,1)); 
  term_2 = (alpha / m) * sum (((X * theta) - y).* X(:,2)); 
  
  theta (1,1) = theta (1,1) - term_1;
  theta (2,1) = theta (2,1) - term_2;


  J_history (i)= Cmpute_Cost_Univariate (theta, X, y);
  end 

end 

theta = GD_Univariate (X, y, theta, alpha, iterations);

% Step_7: Plotting the fitting line
hold on
plot (X(:, 2), X * theta, '-'); 
legend ('Training dataset', 'Linear regression');
hold off

% Step_8: Predict values for population sizes of 35,000
prediction = [1, 3.5] * theta; 
fprintf ('For 35000 population, the predicted profit is %f\n', prediction*10000);

% Step_9: Visualizing J of theta

theta0_vals = linspace(-10, 10, 100);
theta1_vals = linspace(-1, 4, 100);
J_vals = zeros(length(theta0_vals), length(theta1_vals));

for i = 1:length(theta0_vals)
    for j = 1:length(theta1_vals)
	  t = [theta0_vals(i); theta1_vals(j)];
	  J_vals(i,j) = Cmpute_Cost_Univariate(t, X, y);
    end
end

J_vals = J_vals';
figure;
surf(theta0_vals, theta1_vals, J_vals)
xlabel('\theta_0'); ylabel('\theta_1');

figure;

contour(theta0_vals, theta1_vals, J_vals, logspace(-2, 3, 20))
xlabel('\theta_0'); ylabel('\theta_1');
hold on;
plot(theta(1), theta(2), 'rx', 'MarkerSize', 10, 'LineWidth', 2);


