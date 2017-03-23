function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
n = size(theta);  # features
grad = zeros(n);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

% ====================== COST FUNCTION ======================

  [J, grad] = costFunction(theta, X, y);

  J += (lambda/(2*m))*sum(theta(2:end).^2);

% ====================== GRADIENT DESCENT ======================

  # including '0' to avoid - error: costFunctionReg: +=: nonconformant arguments (op1 is 1x28, op2 is 1x27)
  grad += [0 (lambda*theta(2:end)')/m];

  
  #theta = inverse((X')*X + )*X'*y;
  
% =============================================================

end
