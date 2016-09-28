function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

val = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
val_m = size(val, 2);

best = 1;

% for C
for i = 1:val_m 
	% for sigma
	for j = 1:val_m
		% use training set for training
		model = svmTrain(X, y, val(i), @(X, y) gaussianKernel(X, y, val(j)));
		
		% use cross validation for evaluating error 
		predictions = svmPredict(model, Xval);
		score = mean(double(predictions ~= yval));
		if score < best
			best = score;
			C = val(i);
			sigma = val(j);
	end
end

% =========================================================================

end
