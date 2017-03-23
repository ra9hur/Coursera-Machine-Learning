function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
  m = length(y); % number of training examples
  #features = size(X,2); % Number of features

% You need to return the following variables correctly 
  J = 0;
  grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%
% ====================== COST FUNCTION ======================
# h = X*theta;
# error_sq = sum((h-y).^2);

# J = error_sq / (2 * m);

  z= X*theta;
  Hx = sigmoid(z);

  #J = (1/m)*sum(-y'*log(sigmoid(X*theta))-(1-y')*log(1-sigmoid(X*theta)));
  J = (1/m)*sum(sum((-y.*log(Hx)) - (1-y).*(log(1-Hx))));   # since h(x)=g(z);

% ====================== GRADIENT DESCENT ======================
# Some gradient descent settings
# iterations = 1500;
# alpha = 0.01;
%{
  for i = 1:(features)
        grad(i) = (1/m) * sum((Hx'-y).*X(:,i));
  end
%}

  # warning ("off", "Octave:broadcast");

  #grad = (1/m)*(sigmoid(X*theta)-y)'*X;
  grad = (1/m) * sum((Hx-y).*X);

% =============================================================

end
