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

%{
input_layer_size = 2;
hidden_layer_size = 2;
num_labels = 4;
nn_params = [ 1:18 ] / 10;
X = cos([1  2 ; 3  4 ; 5  6]);
y = [4; 2; 3];
lambda = 3;
%}

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
X = [ones(m,1) X];
         
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

% ======== Input Layer
  #fprintf('Theta1 ...\n')
  #size(Theta1)
  #fprintf('X OR a1 ...\n')
  #size(X)
  z2 = X*Theta1';
  a2 = sigmoid(z2);

% ======== Hidden Layer 1

% Add ones to the a2 data matrix
  a2 = [ones(size(a2), 1) a2];
  #fprintf('a2 ...\n')
  #size(a2)
  #fprintf('Theta2 ...\n')
  #size(Theta2)
  z3 = a2*Theta2';

  # Hx is a k dimensional vector
  Hx = sigmoid(z3);   # a3

% ======== Computing Cost

  #  y is being converted to k-dimensional vector
  ytemp = eye(num_labels);
  yk = ytemp(y,:);
  
  # non-regularized
  J = (-1/m)*sum(sum((yk.*log(Hx)) + (1-yk).*(log(1-Hx))));   # since h(x)=g(z);


% ====================== YOUR CODE HERE ======================
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

  d3 = Hx - yk;
  d2 = Theta2'*d3'.*([ones(size(z2), 1) sigmoidGradient(z2)])';
  #d2 = Theta2(2:end,:)'*d3'.*sigmoidGradient(z2)';
  Theta2_grad = (1/m).*d3'*a2;
  Theta1_grad = (1/m).*d2(2:end,:)*X;

% ====================== YOUR CODE HERE ======================
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

  # regularized
  #J += (lambda/(2*m))*(sum(Theta1(2:end).^2) + sum(Theta2(2:end).^2));
  J += (lambda/(2*m))*(sum(sum(Theta1(:,2:end).^2))+sum(sum(Theta2(:,2:end).^2)));

  Theta2_grad += (lambda/m).*[zeros(size(Theta2, 1), 1) Theta2(:,2:end)];
  Theta1_grad += (lambda/m).*[zeros(size(Theta1, 1), 1) Theta1(:,2:end)];

% ========================== END =============================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
