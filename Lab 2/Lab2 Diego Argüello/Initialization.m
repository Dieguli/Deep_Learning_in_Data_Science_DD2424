function [W,b] = Initialization(m,d,K)
mu = 0;
sigma = 0.001;
W1 = mu + randn(m, d)*sigma;
W2 = mu + randn(K, m)*sigma;
b1 = zeros(m, 1);
b2 = zeros(K, 1); 
W= {W1, W2};
b= {b1, b2};
end


