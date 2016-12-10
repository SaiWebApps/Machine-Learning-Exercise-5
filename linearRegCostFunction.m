function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Constants
m = length(y); % number of training examples
n = length(theta); % number of features
jCoefficient = 1 / (2*m);
gradCoefficient = 1 / m;

% Calculate unregularized cost.
h = X * theta;
errors = h - y;
sumOfErrorsSquared = errors' * errors;
jBase = jCoefficient * sumOfErrorsSquared;

% Account for regularization (calculate regularized cost).
theta1n = theta(2:end);
jReg = lambda * sum(theta1n .^ 2);
J = jBase + (jCoefficient * jReg);

% Calculate gradients.
gradBase = gradCoefficient .* (X' * errors);
gradReg = (gradCoefficient * lambda) .* theta1n;
grad = [gradBase(1); gradBase(2:end) + gradReg];

end