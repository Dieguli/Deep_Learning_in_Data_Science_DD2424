function [Wstar, bstar, mu_av, v_av] = MiniBatchGD(X, Y, GDparams, W, b, lambda, rho, alpha, k)

n_batch = GDparams.n_batch;
eta = GDparams.eta;
N = size(X, 2);

for i = 1 : k
    v_W{i} = zeros(size(W{i}));
    v_b{i} = zeros(size(b{i}));
end

for j = 1 : N/n_batch
    j_start = (j - 1)*n_batch + 1;
    j_end = j*n_batch;
    inds = j_start : j_end;
    Xbatch = X(:, inds);
    Ybatch = Y(:, inds);
    [h, S, mu, v] = hiddenlayers(Xbatch, W, b, k);
    P = EvaluateClassifier(h, W, b);  
    for i = 1 : k - 1
        if j == 1
            mu_av = mu;
            v_av = v;
        else
            mu_av{i} = alpha*mu_av{i} + (1 - alpha)*mu{i};
            v_av{i} = alpha*v_av{i} + (1 - alpha)*v{i};
        end
    end
    
    [grad_W, grad_b] = ComputeGradients(Xbatch, Ybatch, P, h, S, W, lambda, k, mu, v);
    
    for i = 1 : k
        v_W{i} = rho*v_W{i} + eta*grad_W{i};
        v_b{i} = rho*v_b{i} + eta*grad_b{i};
        W{i} = W{i} - v_W{i};
        b{i} = b{i} - v_b{i};
    end
end

Wstar = W;
bstar = b;

end
