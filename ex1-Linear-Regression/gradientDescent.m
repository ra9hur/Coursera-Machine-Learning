function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
features = size(X,2); % Number of features
J_history = zeros(num_iters, 1);
temp = zeros(2, 1); % initialize temp theta parameters

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
  
  for i = 1:features
	    H = X * theta;
	
      if i==1,
	      temp(i) = theta(i) - alpha * (1/m) * sum(H-y);
	    else
        temp(i) = theta(i) - alpha * (1/m) * sum((H-y).*X(:,i));
      end
      
  end

  theta = temp;
  % theta(1) = temp0;
	% theta(2) = temp1;
  % ============================================================
	
  % Save the cost J in every iteration    
  J_history(iter) = computeCost(X, y, theta);

end

end

#{ Not getting into this loop, commenting this out
	if (J_history(iter) == 0),
		J_history(iter-1)
		J_history(iter)
		iter
		return;
end #}