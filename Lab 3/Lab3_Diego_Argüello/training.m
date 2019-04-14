function [W, b, Jtr, Jva, check, mu_av, v_av] = training(Xtr, Ytr, Xva, Yva, GDparams, lambda, k, m)

d = size(Xtr, 1);
K = size(Ytr, 1);
M = [d, m, K];
sigma = 0.001;
[W, b] = initialize(M, sigma);
Jtr = zeros(1, GDparams.n_epochs);
Jva = zeros(1, GDparams.n_epochs);
decay_rate = 0.8;
r = 0.9;
alpha = 0.99;
check = 0;

for i = 1 : GDparams.n_epochs
    [W, b, mu_av, v_av] = MiniBatchGD(Xtr, Ytr, GDparams, W, b, lambda, r, alpha, k);
%     [W, b] = MiniBatchGD(Xtr, Ytr, GDparams, W, b, lambda, rho, alpha, k);
    Jtr(i) = ComputeCost(Xtr, Ytr, W, b, lambda, k, mu_av, v_av);
    Jva(i) = ComputeCost(Xva, Yva, W, b, lambda, k, mu_av, v_av);
% Jtr(i) = ComputeCost(Xtr, Ytr, W, b, lambda, k);
%     Jva(i) = ComputeCost(Xva, Yva, W, b, lambda, k);
    if Jtr(i) > 3*Jtr(1)
        check = 1;
        break;
    end
    GDparams.eta = decay_rate*GDparams.eta;
end

end
