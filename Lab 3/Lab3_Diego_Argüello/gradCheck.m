function gradCheck(batch_size, X, Y, lambda, W, b, delta, k)

[ngrad_b, ngrad_W] = ComputeGradsNumSlow(X(:, 1 : batch_size), ...
    Y(:, 1 : batch_size), W, b, lambda, delta);

% mu_av=[];
% v_av = [];
[h, S] = hiddenlayers(X(:, 1 : batch_size), W, b, k,mu_av, v_av);
P = EvaluateClassifier(h, W, b);
%  [grad_W, grad_b] = CompGrad(X, Y, P, h, S, W, lambda, k);
[grad_W, grad_b] = ComputeGradients(X(:, 1 : batch_size), ...
    Y(:, 1 : batch_size), P, h, S, W, lambda, k);
eps = 0.001;
for i = 1 : length(W)
    gradcheck_bi = sum(abs(ngrad_b{i} - grad_b{i})/max(eps, sum(abs(ngrad_b{i}) + abs(grad_b{i}))));
    gradcheck_Wi = sum(sum(abs(ngrad_W{i} - grad_W{i})/max(eps, sum(sum(abs(ngrad_W{i}) + abs(grad_W{i}))))));
    disp(['error grad_W' num2str(i) ': ' num2str(gradcheck_Wi)])
    disp(['error grad_b' num2str(i) ': ' num2str(gradcheck_bi)])
   
end

end