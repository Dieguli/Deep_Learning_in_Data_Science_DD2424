function [Wout, bout] = MiniBatchGD_old(X, Y, GDparams, W, b, lambda)
n_epochs = GDparams.n_epochs;
n_batch = GDparams.n_batch;
eta = GDparams.eta;
[M,N] = size(X);
% J = zeros (1,(N/n_batch));
for j = 1 : N/n_batch
%     Code given in the assignemnt instructions
    j_start = (j - 1)*n_batch + 1;
    j_end = j*n_batch;
    inds = j_start : j_end;
    Xbatch = X(:, j_start : j_end);
    Ybatch = Y(:, j_start : j_end);

    H = firstLayer(Xbatch, W, b); 
    P = EvaluateClassifier(H, W, b);
    [grad_W, grad_b] = ComputeGradients(Xbatch, Ybatch, P, H, W, b, lambda);
    
    W{1} = W{1} - eta*grad_W{1};
    W{2} = W{2} - eta*grad_W{2};
    b{1} = b{1} - eta*grad_b{1};
    b{2} = b{2} - eta*grad_b{2};
    
end

Wout = W;
bout = b;


end

