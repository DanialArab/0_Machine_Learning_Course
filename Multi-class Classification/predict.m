function p = predict(Theta1, Theta2, X)

m = size(X, 1);
num_labels = size(Theta2, 1);

p = zeros(size(X, 1), 1);


activation1 = [ones(m, 1) X];
z_2 = activation1 * Theta1';
activation2 = [ones(size(z_2, 1), 1) sigmoid(z_2)];
z_3 = activation2 * Theta2';
activation3 = sigmoid(z_3);

[p_max, p] = max(activation3, [], 2);


end
