function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

    % firstly, our parameter is a (n+1) * 1 vector, in this case 2 * 1
    % the prediction is equal to X * theta, and the difference between
    % the prediction and y is a vector of shape (97, 1) in this case.

    % we need to vectorize, such that the resulting vector of the 
    % derivatives of the loss function matches the shape of theta

    % our data matrix X is of shape (97, 2); and the error vector, resulted
    % by (prediction - y) is of shape (97, 1)
    % then, we use element-wise multiplication to strech our feature vector
    % x = [x1, x2] where x1 = 1, x2 = population, by that (pred - y)
    % the resulting vector after summation is of shape (1, 2), then we take
    % the transpose such that it can be used to update the theta
    theta = theta - alpha*(1/m)*sum(((X*theta - y) .* X))'


    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
