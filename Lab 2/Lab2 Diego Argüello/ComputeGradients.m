function [grad_W, grad_b] = ComputeGradients(X, Y, P, H, W, b, lambda)
W1 = W{1};
W2 = W{2};
b1 = b{1};
b2 = b{2};
grad_W1 = zeros(size(W1));
grad_W2 = zeros(size(W2));
grad_b1 = zeros(size(b1));
grad_b2 = zeros(size(b2));

for i = 1 : size(X, 2)
    p = P(:, i);
    h = H(:, i);
    y = Y(:, i);
    x = X(:, i);
    g = -y'*(diag(p) - p*p')/(y'*p);
    grad_b2 = grad_b2 + g';
    grad_W2 = grad_W2 + g'*h';
    g = g*W2;
    h(find(h > 0)) = 1;
    g = g*diag(h);
    grad_b1 = grad_b1 + g';
    grad_W1 = grad_W1 + g'*x';   
end

grad_W1 = 2*lambda*W1 + grad_W1/size(X, 2);
grad_W2 = 2*lambda*W2 + grad_W2/size(X, 2);
grad_b1 = grad_b1/size(X, 2);
grad_b2 = grad_b2/size(X, 2);
grad_W = {grad_W1, grad_W2}; 
grad_b = {grad_b1, grad_b2};

end