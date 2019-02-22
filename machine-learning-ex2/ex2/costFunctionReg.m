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

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
J = 1/m *(-y' * log(sigmoid(X*theta)) - (1-y)' * log(1 - sigmoid(X*theta))) + (lambda/(2*m)) *sum(theta(2:size(theta,1),:) .^2);

%theta(1)和其他值的计算方法是不一样的。两个需要分开计算。
grad1 = 1/m *X'(1,:) * (sigmoid(X*theta) - y);
grad2 = 1/m * [zeros(1,size(X',2)); X'(2:size(X',1),:)] *(sigmoid(X*theta) - y) + (lambda/m) * theta;
grad = [grad1;grad2(2:size(grad2,1),:)];




% =============================================================

end
