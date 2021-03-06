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
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% feedforward operation
% 1. add ones' vector to X
X = [ones(m, 1) X]; % (5000, 401)

z2 = Theta1*X'; % (25, 401)*(401, 5000) 
a2 = sigmoid(z2); % (25, 5000)
a2 = [ones(m, 1) a2']; % (26, 5000)

z3 = Theta2*a2'; % (10, 5000)
a3 = sigmoid(z3); 

% 2. implement the cost function (without regularization)
Y = zeros(num_labels, m);
% convert labels y to matrix Y, each row is the label vector y
for i=1:m,
    Y(y(i), i) = 1;
end;
% after that, we can vectorize the implementation of the summations
% convert Y to matrix of shape (num_examples, num_labels);
% as our prediction 'a3' is of shape (5000, num_labels); 
% by doing element-wise multiplication, we can get the specific class
% for each example (e.g.) the prediction value of the first example
% can be [0.5, 0.4, 0.3] (assume we have 3 class), and the first row
% of Y is [0, 1, 0], assume the class of this example is 2. Then,
% Y .* log(a3) -> [0, log(0.4), 0], and [log(1-0.5), 0, log(1-0.3)]
% we first sum over the columns, get a vector of shape (5000, 1)
% after that we sum over all the examples
J = (1/m)* sum ( sum ( -Y .* log(a3) - (1-Y) .* log(1-a3) ));

% now add regularization term into the cost
% the regularization just basically the sum of theta's square without
% taking account of the first column of each theta matrix
reg_loss = (lambda/(2*m)) * ( sum(sum(Theta1(:, 2:end).^2)) + sum(sum(Theta2(:, 2:end).^2)));

J = J + reg_loss;

% 3. backpropagation

% 3.1. compute delta for the network
for i=1:m,
    a1 = X(i, :)'; % (401, 1)
    % forward propagation
    z2 = Theta1*a1; % (25, 1) 
    a2 = sigmoid(z2);  % (25, 1)
    a2 = [1; a2]; % (26, 1)
    z3 = Theta2*a2; % (10, 1)
    a3 = sigmoid(z3); % (10, 1)
    y = Y(:, i);

    delta_3 = a3 - y; % (10, 1)
    z2 = [1; z2]; % ( 26, 1)
    delta_2 = (Theta2'*delta_3) .* sigmoidGradient(z2); % (26,10)(10,1)

    delta_2 = delta_2(2:end);

    Theta2_grad = Theta2_grad + delta_3*a2'; 
    Theta1_grad = Theta1_grad + delta_2*a1';

end;

Theta2_grad = (1/m) * Theta2_grad; % (10*26)
Theta1_grad = (1/m) * Theta1_grad; % (25*401)

% add regularization to gradient

Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) + (lambda/m)*(Theta2(:,2:end));
Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) + (lambda/m)*(Theta1(:,2:end));

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
