function [W, b, CostTraining, CostValidation, flag] = FindingParameters(trainX, trainY, valX, valY, GDparams, lambda, m)

d = size(trainX,1);
K = size (trainY,1);
mu = 0;
sigma = 0.001;
W1 = mu + randn(m, d)*sigma;
W2 = mu + randn(K, m)*sigma;
b1 = zeros(m, 1);
b2 = zeros(K, 1);
 W{1} = W1;
 W{2} = W2;
b{1}= b1;
b{2}= b2 ;
CostTraining = zeros(1, GDparams.n_epochs);
CostValidation = zeros(1, GDparams.n_epochs);
decay_rate = 0.95;
r = 0.9;
flag = 0;
for i = 1 : GDparams.n_epochs
    CostTraining(i) = ComputeCost(trainX, trainY, W, b, lambda);
    CostValidation(i) = ComputeCost(valX, valY, W, b, lambda);
    if CostTraining(i) > 3*CostTraining(1)
        flag = 1;
        break;
    end
    [W, b] = MiniBatchGD(trainX, trainY, GDparams, W, b, lambda, r);
    GDparams.eta = decay_rate*GDparams.eta;
end

end