function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% a1 is (m x n), where 'n' is the number of features including the bias unit
a1 = [ones(m, 1) X];

% Theta1 is (h x n) where 'h' is the number of hidden units
% a2 is (m x (h + 1))
z2 = Theta1 * a1.';
a2 = [ones(m, 1) sigmoid(z2.')];

% Theta2 is (c x (h + 1)), where 'c' is the number of labels.
% a3 is (m x c)
z3 = a2 * Theta2.';
a3 = sigmoid(z3);

% p is a vector of size (m x 1)
[dummy p] = max(a3, [], 2);


% =========================================================================


end
