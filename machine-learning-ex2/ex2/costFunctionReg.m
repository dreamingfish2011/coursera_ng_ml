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
n = size(theta);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

 
%step1:compute hx
hx = X*theta;
 
%step2:compute h(hx)
h = sigmoid(hx);
 
%step3:compute cost function's sum part
J = -(1/m) * [y'* log(h) + (1-y')*log(1-h)] + (lambda/(2*m)) * (theta(2:n)' * theta(2:n)) ;
%step4:compute gradient
grad(1) = (h-y)'*X(:,1)/m;  %%sum(h-y)/m
grad(2:n) = (1/m)*[X(:,2:n)' * (h-y) + lambda*theta(2:n)]; 



% =============================================================

end
