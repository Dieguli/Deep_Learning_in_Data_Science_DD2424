function J = ComputeCost(X, Y, W, b, lambda)
% J is a scalar corresponding to the sum of the loss of the network’s predictions 
% for the images in X relative to the ground truth labels and the regularization term on W.

W1 = W{1};
W2 = W{2};

H = firstLayer(X, W, b);
P = EvaluateClassifier(H, W, b);
J1 = sum(diag(-log(Y'*P)))/size(X, 2);
J2 = lambda*(sum(sum(W1.^2)) + sum(sum(W2.^2)));
J = J1 + J2;

end