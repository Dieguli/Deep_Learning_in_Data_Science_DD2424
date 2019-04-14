function [grad_W, grad_b] = ComputeGradients(X, Y, P, W, lambda)
[Rx, Cx] = size(X);
[Rw, Cw] = size(W);
grad_W = zeros(Rw,Cw);
grad_b = zeros(Rw, 1);

for i = 1 : Cx
    p = P(:, i);
    y = Y(:, i);
    x= X(:, i);
    g = -y'*(diag(p) - p*p')/(y'*p);
    grad_b = grad_b + g';
    grad_W = grad_W + g'*x';
end

grad_b = grad_b/Cx;
grad_W = 2*lambda*W + grad_W/Cx;

end

