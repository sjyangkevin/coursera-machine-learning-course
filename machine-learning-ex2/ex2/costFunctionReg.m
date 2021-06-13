function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

# the data loss part is the same, and we add a regularization loss
# as we are not penalize theta_0, so, we skip theta_0;

# the gradient for the data loss is the same, we just need to
# add the regularization part. for the gradient. we also need
# to skip theta_0, as we not gonna penalize that
J = (1/m)*sum(-y' * log(sigmoid(X*theta)) - (1-y)' * log(1-sigmoid(X*theta))) + (lambda/(2*m))*sum(theta(2:end).^2);
grad = (1/m) * (X' * (sigmoid(X*theta) - y));
grad = [grad(1); grad(2:end) + (lambda/m) * theta(2:end)];

% =============================================================

end
