function J = ComputeCost(X, Y, W, b, lambda)
% J is a scalar corresponding to the sum of the loss of the network’s predictions 
% for the images in X relative to the ground truth labels and the regularization term on W.
[R,D] = size(X);
P = EvaluateClassifier(X, W, b);
J = sum(diag(-log(Y'*P)))/D + lambda*sum(sum(W.^2)); 
end

