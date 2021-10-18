clear; close all; clc;

% Step_1: loading the data

data = load ('ex2data1.txt');
X = data (:, 1:2);
y = data (:, 3);

% Step_2: visualizing data to understand them

positive = find (y ==1); 
negative = find (y ==0);

figure 
plot(X(positive, 1), X (positive, 2), 'k+', 'LineWidth', 2, 'MarkerSize', 7);
hold on 
plot(X(negative, 1), X (negative, 2), 'ko', 'MarkerFaceColor', 'y', 'MarkerSize', 7);

xlabel ('Exam one score')
ylabel('Exam two score')
legend('Admitted', 'Not admitted'); 


% Step_3: Defining the logistic regression hypothesis function, which is sigmoid function

function g = sigmoid (z)
  g = 1./(1 + exp(-z));
end

% Step_4: Cost function and gradient 

function [J, grad] = CostFunction (theta, X, y)
  m = size(y, 1);   
  J = zeros (m, 1);
  
  J = 1 / m .* (- y' * log(sigmoid (X*theta)) - (1 - y)' * log(1 - sigmoid(X*theta))); % vectorized implementation
  grad = 1 / m * X' * (sigmoid (X* theta) - y); % vectorized implementation
  
end

% Step_5: initializing fitting parameters
n = size(X, 2) + 1; 
initial_theta = zeros(n, 1); 

% Step_6: adding one columns to X for the theta_0
m = size(y, 1);   
X = [ones(m, 1) X]; 


% Step_7: Calculating cost and gradient through calling CostFunction 

[cost, grad] = CostFunction (initial_theta, X, y)

% Step_8: Learning parameters unig fminunc
options = optimset('GradObj', 'on', 'MaxIter', 400);

[theta, cost] = ...
	fminunc(@(t)(costFunction(t, X, y)), initial_theta, options);
  
fprintf ('The cost calculated from fminunc is %f\n', cost)';
fprintf('Expected cost (approx): 0.203\n'); 
fprintf('The thetas calculated from fminunc are: \n');
fprintf(' %f \n', theta);
fprintf('Expected theta (approx):\n');
fprintf(' -25.161\n 0.206\n 0.201\n');


% Step_9: Plotting the decision boundary
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
xlabel('Exam 1 score')
ylabel('Exam 2 score')
legend('Admitted', 'Not admitted')
hold off;

% Step_10: Prediction and estimating accuracies 

function p = predict(theta, X)

m = size(X, 1); 

p = zeros(m, 1);

for i = 1: m
  if sigmoid (X * theta) (i) >= 0.5 
    p (i) = 1;
  else
    p (i) = 0;
end

end

end 

prob = sigmoid([1 45 85] * theta);
fprintf(['For a student with scores 45 and 85, we predict an admission ' ...
         'probability of %f\n'], prob);
fprintf('Expected value: 0.775 +/- 0.002\n\n');

% Compute accuracy on our training set
p = predict(theta, X);

fprintf('Train Accuracy: %f\n', mean(double(p == y)) * 100);
fprintf('Expected accuracy (approx): 89.0\n');
fprintf('\n');
