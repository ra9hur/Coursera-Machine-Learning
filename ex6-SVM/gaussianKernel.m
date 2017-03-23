function f = gaussianKernel(x1, x2, sigma)
%RBFKERNEL returns a radial basis function kernel between x1 and x2
%   sim = gaussianKernel(x1, x2) returns a gaussian kernel between x1 and x2
%   and returns the value in sim

% Ensure that x1 and x2 are column vectors
x1 = x1(:); x2 = x2(:);

% You need to return the following variables correctly.
f = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the similarity between x1
%               and x2 computed using a Gaussian kernel with bandwidth
%               sigma
%
%

  ttl_dist = 0;
%{
  for i = 1:length(x1)
    ttl_dist += sum((x1(i) .- x2))^2;
  endfor
%}

#  ttl_dist = sum(x1-x2).^2;         # variation in summation brackets
  
#  sim = exp(-ttl_dist/(2*sigma^2));

  # Do perform feature scaling if required to get realistic values
  f = exp(-sum((x1-x2).^2)/(2*sigma^2));

  

% =============================================================
    
end
