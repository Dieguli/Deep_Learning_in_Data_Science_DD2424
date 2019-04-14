function [Wout, bout] = MiniBatchGD(X, Y, GDparams, W, b, lambda, r)
%Momentum added
n_batch = GDparams.n_batch;
eta = GDparams.eta;
% decay_rate = 0.95;
N = size(X, 2);  %number of images 
Wv = {zeros(size(W{1})), zeros(size(W{2}))};
bv = {zeros(size(b{1})), zeros(size(b{2}))};
% As the images in the CIFAR-10 dataset are in random order, the easiest to 
% generate each mini-batch is to just run through the images sequentially. 
% Let n _atch be the number of images in a mini-batch. Then for one epoch 
% (a complete run through all the training images), you can generate the set 
% of mini-batches with this snippet of code:

for j = 1 : N/n_batch
    
    j_start = (j - 1)*n_batch + 1;
    j_end = j*n_batch;
    inds = j_start : j_end; 
    Xbatch = X(:, inds);
    Ybatch = Y(:, inds);
    H = firstLayer(Xbatch, W, b);
    P = EvaluateClassifier(H, W, b);
    [grad_W, grad_b] = ComputeGradients(Xbatch, Ybatch, P, H, W, b, lambda);
    Wv{1} = r*Wv{1} + eta*grad_W{1};
    Wv{2} = r*Wv{2} + eta*grad_W{2};
    bv{1} = r*bv{1} + eta*grad_b{1};
    bv{2} = r*bv{2} + eta*grad_b{2};
    W{1} = W{1} - Wv{1};
    W{2} = W{2} - Wv{2};
    b{1} = b{1} - bv{1};
    b{2} = b{2} - bv{2};
end

Wout = W;
bout = b;

end