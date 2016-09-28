function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%

% Forward computation for htheta(X)
a1 = [ones(size(X, 1), 1) X];
a2 = sigmoid(a1 * Theta1.');
a2 = [ones(size(a2, 1), 1) a2];
htheta = sigmoid(a2 * Theta2.'); % 5000 x 10

% Converting y to vectors representing each digit 
% Generates a 10 x 10 identity matrix
y_label = eye(num_labels);

% Uses logicals in array indexing to create vectors
y_vector = y_label(y,:); % 5000 x 10

% Cost function for neural network without regularisation
% One sums for m and k each, otherwise it'll just be a 5000 x 10 matrix
J = (sum(sum(-y_vector .* log(htheta) - (1 - y_vector) .* log(1 - htheta)))) / m;

% Regularisation of Theta1 and Theta2 without their first columns (you don't
% regularise the bias)
rgl = (lambda / (2 * m)) * (sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2)));
J = J + rgl;

% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%

d3 = htheta - y_vector; % 5000 x 10
d2 = (d3 * Theta2(:,2:end)) .* sigmoidGradient(a1 * Theta1.'); % 5000 x 25

% Accumulate unregularised gradients
D1 = d2.' * a1; % 25 x 401
D2 = d3.' * a2; % 10 x 26

% Final steps
Theta1_grad = D1 / m;
Theta2_grad = D2 / m;

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% Copying Theta1 and Theta2 matrices and removing bias values (set to 0)
T1greg = Theta1;
T1greg(:,1) = 0;
T2greg = Theta2;
T2greg(:,1) = 0;

% Add regularisation
T1greg = T1greg * (lambda / m);
T2greg = T2greg * (lambda / m); 

% Add regularisation and gradient from part 2
Theta1_grad = Theta1_grad + T1greg;
Theta2_grad = Theta2_grad + T2greg;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
