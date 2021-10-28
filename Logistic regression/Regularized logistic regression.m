clear ; close all; clc

% Part 1: Regularized Logistic Regression

% Step_1: loading the data

data = load ('ex2data2.txt');
X = data (:, 1:2);
y = data (:, 3);

% Step_2: visualizing data to understand them

positive = find (y ==1); 
negative = find (y ==0);

figure 
plot(X(positive, 1), X (positive, 2), 'k+', 'LineWidth', 2, 'MarkerSize', 7);
hold on 
plot(X(negative, 1), X (negative, 2), 'ko', 'MarkerFaceColor', 'y', 'MarkerSize', 7);

xlabel('Microchip Test 1')
ylabel('Microchip Test 2')
legend('y=1', 'y=0'); 
hold off; 

% Step_3: Featuring Mapping 

function out = mapFeature(X1, X2)
degree = 6;
out = ones(size(X1(:,1)));
for i = 1:degree
    for j = 0:i
        out(:, end+1) = (X1.^(i-j)).*(X2.^j);
    end
end

end

X = mapFeature(X(:,1), X(:,2));

% Step_4: Initialize fitting parameters and set regularization parameter lambda to 1
initial_theta = zeros(size(X, 2), 1);
lambda = 1;

% Step_5: Defining the logistic regression hypothesis function, which is sigmoid function

function g = sigmoid (z)
  g = 1./(1 + exp(-z));
end


function [J, grad] = costFunctionReg(theta, X, y, lambda)

m = length(y);

J = 0;
grad = zeros(size(theta));

n = length(theta); 

for i = 1: m
  J = 1 / m .*( - y' * log(sigmoid(X* theta)) - (1 - y)' * log(1- sigmoid(X*theta))); 
  for j = 2:n
    J = J + lambda / 2 / m *sum(theta(j, 1).^2); 
  end
end

for i = 1:m
  grad (1,1)= 1 / m * sum ((sigmoid(X * theta) - y) .* X(:,1)); 
  for j = 2 : n
    grad (j,1)= 1 / m * sum((sigmoid (X * theta) - y).*X(:,j)) + lambda / m * theta(j, 1); 
  end
end


end


[cost, grad] = costFunctionReg(initial_theta, X, y, lambda);

fprintf('Cost at initial theta (zeros): %f\n', cost);
fprintf('Expected cost (approx): 0.693\n');
fprintf('Gradient at initial theta (zeros) - first five values only:\n');
fprintf(' %f \n', grad(1:5));
fprintf('Expected gradients (approx) - first five values only:\n');
fprintf(' 0.0085\n 0.0188\n 0.0001\n 0.0503\n 0.0115\n'); 

% Compute and display cost and gradient, with all-ones theta and lambda = 10
test_theta = ones(size(X,2),1);
[cost, grad] = costFunctionReg(test_theta, X, y, 10);

fprintf('\nCost at test theta (with lambda = 10): %f\n', cost);
fprintf('Expected cost (approx): 3.16\n');
fprintf('Gradient at test theta - first five values only:\n');
fprintf(' %f \n', grad(1:5));
fprintf('Expected gradients (approx) - first five values only:\n');
fprintf(' 0.3460\n 0.1614\n 0.1948\n 0.2269\n 0.0922\n');

% Part 2: Regularization and Accuracies 

% Step_1: Initialize fitting parameters and set regularization parameter lambda to 1
initial_theta = zeros(size(X, 2), 1);
lambda = 1;

% Step_2: Set Options
options = optimset('GradObj', 'on', 'MaxIter', 400);

% Step_3: Optimize
[theta, J, exit_flag] = ...
	fminunc(@(t)(costFunctionReg(t, X, y, lambda)), initial_theta, options);

% Plot Boundary

function plotDecisionBoundary(theta, X, y)

plotData(X(:,2:3), y);
hold on

if size(X, 2) <= 3
    plot_x = [min(X(:,2))-2,  max(X(:,2))+2];

    
    plot_y = (-1./theta(3)).*(theta(2).*plot_x + theta(1));

    
    plot(plot_x, plot_y)
    
    legend('Admitted', 'Not admitted', 'Decision Boundary')
    axis([30, 100, 30, 100])
else
    
    u = linspace(-1, 1.5, 50);
    v = linspace(-1, 1.5, 50);

    z = zeros(length(u), length(v));
    
    for i = 1:length(u)
        for j = 1:length(v)
            z(i,j) = mapFeature(u(i), v(j))*theta;
        end
    end
    z = z'; 

   
    contour(u, v, z, [0, 0], 'LineWidth', 2)
end
hold off

end

plotDecisionBoundary(theta, X, y);
hold on;
title(sprintf('lambda = %g', lambda))

% Labels and Legend
xlabel('Microchip Test 1')
ylabel('Microchip Test 2')

legend('y = 1', 'y = 0', 'Decision boundary')
hold off;

% Compute accuracy on our training set
p = predict(theta, X);

fprintf('Train Accuracy: %f\n', mean(double(p == y)) * 100);
fprintf('Expected accuracy (with lambda = 1): 83.1 (approx)\n');



