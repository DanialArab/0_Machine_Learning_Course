% Step_1: loading the data, which is in form of a native OCTAVE matrix format,
% the matrix will be already named and there is no need to assign names to them

clear; close all; clc;

load ('ex3data1.mat');

% the grayscale images are 20 by 20 pixels, which were unrolled as 400 array and since we have
% 5000 examples in the training examples the X matrix is 5000 by 400
% in y, zero is labelled as 10 while the other digits, 1 to 9, are labelled consistently

% Step_2: visualizing the random subset of the data
function [h, display_array] = displayData(X, example_width)

if ~exist('example_width', 'var') || isempty(example_width) 
	example_width = round(sqrt(size(X, 2)));
end

colormap(gray);

[m n] = size(X);
example_height = (n / example_width);

display_rows = floor(sqrt(m));
display_cols = ceil(m / display_rows);

pad = 1;

display_array = - ones(pad + display_rows * (example_height + pad), ...
                       pad + display_cols * (example_width + pad));

curr_ex = 1;
for j = 1:display_rows
	for i = 1:display_cols
		if curr_ex > m, 
			break; 
		end
		max_val = max(abs(X(curr_ex, :)));
		display_array(pad + (j - 1) * (example_height + pad) + (1:example_height), ...
		              pad + (i - 1) * (example_width + pad) + (1:example_width)) = ...
						reshape(X(curr_ex, :), example_height, example_width) / max_val;
		curr_ex = curr_ex + 1;
	end
	if curr_ex > m, 
		break; 
	end
end

h = imagesc(display_array, [-1 1]);

axis image off

drawnow;

end

input_layer_size  = 400;  
num_labels = 10;      
m = size(X, 1);

rand_indices = randperm(m);
sel = X(rand_indices(1:100), :);

displayData(sel);

% Step_3: Vectorizing the cost function of logistic regression 

function [J, grad] = lrCostFunction(theta, X, y, lambda)

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

% compute gradient
for i = 1:m
  grad (1,1)= 1 / m * sum ((sigmoid(X * theta) - y) .* X(:,1)); 
  for j = 2 : n
    grad (j,1)= 1 / m * sum((sigmoid (X * theta) - y).*X(:,j)) + lambda / m * theta(j, 1); 
  end
end


grad = grad(:);

end

% Testing logistic regression cost function

theta_t = [-2; -1; 1; 2];
X_t = [ones(5,1) reshape(1:15,5,3)/10];
y_t = ([1;0;1;0;1] >= 0.5);
lambda_t = 3;
[J grad] = lrCostFunction(theta_t, X_t, y_t, lambda_t);

fprintf('\nCost: %f\n', J);
fprintf('Expected cost: 2.534819\n');
fprintf('Gradients:\n');
fprintf(' %f \n', grad);
fprintf('Expected gradients:\n');
fprintf(' 0.146561\n -0.548558\n 0.724722\n 1.398003\n');

% Step_4: One-vs-All classification by training multiple regularized logistic regression classifiers
% fmincg works similarly to fminunc, but is more more efficient for dealing with a large number of parameters.


function [all_theta] = oneVsAll(X, y, num_labels, lambda)

m = size(X, 1);
n = size(X, 2);

all_theta = zeros(num_labels, n + 1);

X = [ones(m, 1) X];


for i = 1:num_labels
  initial_theta = zeros(n+1, 1);
  options = optimset('GradObj', 'on', 'MaxIter', 50);
  [theta] = fmincg(@(t)(lrCostFunction(t, X, (y==i), lambda)), ...
                   initial_theta, options);
  all_theta(i, :) = theta';
end


end


lambda = 0.1;
[all_theta] = oneVsAll(X, y, num_labels, lambda);

% Step_5: One-vs-all Prediction

function p = predictOneVsAll(all_theta, X)

m = size(X, 1);
num_labels = size(all_theta, 1);

p = zeros(size(X, 1), 1);

X = [ones(m, 1) X];

predict = sigmoid(X*all_theta');
[predict_max, index_max] = max(predict, [], 2);
p = index_max;

end

pred = predictOneVsAll(all_theta, X);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);
