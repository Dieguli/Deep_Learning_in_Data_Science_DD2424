function [Wstar, bstar] = MiniBatchGD(X, Y, GDparams, W, b, lambda)
n_epochs = GDparams.n_epochs;
n_batch = GDparams.n_batch;
eta = GDparams.eta;
[M,N] = size(X);

for j = 1 : N/n_batch
%     Code given in the assignemnt instructions
    j_start = (j - 1)*n_batch + 1;
    j_end = j*n_batch;
    inds = j_start : j_end;
    Xbatch = X(:, j_start : j_end);
    Ybatch = Y(:, j_start : j_end);
%     Gradient computation
    P = EvaluateClassifier(Xbatch, W, b);
    [grad_W, grad_b] = ComputeGradients(Xbatch, Ybatch, P, W, lambda);
%     Learnig 
    W = W - eta*grad_W;
    b = b - eta*grad_b;
end

Wstar = W;
bstar = b;

end