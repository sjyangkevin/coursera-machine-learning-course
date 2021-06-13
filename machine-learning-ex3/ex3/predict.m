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

% add 1's to the matrix
X = [ones(m, 1) X];

% currently, X + bias unit, is of shape (m, n + 1), and Theta1 is of shape
% (?, n + 1); by transpose Theta1, we get a weight matrix of shape (n + 1, ?);
% by multiplying it with X, we got a matrix of shape (m, ?); each row is an 
% training example, and the elements in the row is the output of the actiation
% function from each unit in layer 2. similar for layer 3.

% we make the Theta_1 transpose, because by doing this, each column in Theta1
% is corresponding to an activation unit. by multiplying with matrix X, we got
% a inner product for each row in X and each column in Theta1, which is z.
z2 = X * Theta1';
g2 = sigmoid(z2);
g2 = [ones(size(g2, 1), 1) g2];
z3 = g2 * Theta2';
g3 = sigmoid(z3);

[_, p] = max(g3, [], 2);






% =========================================================================


end
