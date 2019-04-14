function [acc, k_star] = ComputeAccuracy(X, y, W, b, k, mu_av, v_av)


%no bn
%     h = hiddenlayers(X, W, b, k);

    h = hiddenlayers(X, W, b, k, mu_av, v_av);
    
P = EvaluateClassifier(h, W, b);
[~, k_star] = max(P);
acc = length(find(y - k_star == 0))/length(y);
K = length(unique(k_star));
N = length(k_star);
k_star = zeros(K, N);
for i = 1 : N
    k_star(y(i), i) = 1; 
    
end 

end
