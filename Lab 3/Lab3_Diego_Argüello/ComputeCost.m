function J = ComputeCost(X, Y, W, b, lambda, k, mu_av, v_av)


%no bn
   % h = hiddenlayers(X, W, b, k);

    h = hiddenlayers(X, W, b, k, mu_av, v_av);

P = EvaluateClassifier(h, W, b);
J1 = sum(diag(-log(Y'*P)))/size(X, 2);

J2 = 0;
for i = 1 : length(W)
    temp = W{i}.^2;
    J2 = J2 + lambda*sum(temp(:));
end

J = J1 + J2;

end