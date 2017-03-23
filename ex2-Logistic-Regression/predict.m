function p = predict(theta, X)
%PREDICT Predict whether the label is 0 or 1 using learned logistic 
%regression parameters theta
%   p = PREDICT(theta, X) computes the predictions for X using a 
%   threshold at 0.5 (i.e., if sigmoid(theta'*x) >= 0.5, predict 1)

m = size(X, 1); % Number of training examples

% You need to return the following variables correctly
p = zeros(m, 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned logistic regression parameters. 
%               You should set p to a vector of 0's and 1's
%

# g(z) >= 0.5, if z >= 0
# y = 1, when theta'*X >= 0

# Theta
# -25.161272
# 0.206233
# 0.201470
# -25*X0 + 0.2*X1 + 0.2*X2 >= 0   OR    Solving, X1 + X2 >= 125

#  p = arrayfun(@(t)(ge(t, 0.5)), sigmoid(X*theta));


  z= X*theta;
  Hx = sigmoid(z);
  
  p = (Hx >= 0.5);


% =========================================================================


end
