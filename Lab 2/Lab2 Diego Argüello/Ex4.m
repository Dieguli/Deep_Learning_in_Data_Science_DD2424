close all;
clear all;
addpath cifar-10-matlab/cifar-10-batches-mat; 
[trainX, trainY, trainy] = LoadBatch('data_batch_1.mat');
[valX, valY, valy] = LoadBatch('data_batch_2.mat');
[testX, testY, testy] = LoadBatch('test_batch.mat');
mean_trainX = mean(trainX, 2);
trainX = trainX- repmat(mean_trainX, [1, size(trainX, 2)]);
valX = valX - repmat(mean_trainX, [1, size(valX, 2)]);
testX = testX - repmat(mean_trainX, [1, size(testX, 2)]);
% lambda = 0.000001;
m = 50;
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
% First round epochs =10; second round epochs=15
% GDparams.n_epochs = 15;
% GDparams.n_batch = 100;
% costTraining=zeros(1,GDparams.n_epochs);
% costValidation = zeros(1, GDparams.n_epochs);
% % decay_rate = 0.95; 
% r = 0.9;
% % ------------ Manual check of the range of the learning rate -------------------------------------------------------
% % % emax = 0.3; Nothing happens
% % % If the learning rate is too small, then after each epoch you will find 
% % % that the training loss barely changes. While if the learning rate is too large,
% % % then learning is unstable and you will probably get NaNs and/or very high loss values
% e = 0.075;
% % GDparams.eta = e;
% % %Training by hand
% % for i = 1:GDparams.n_epochs
% % [W_i,b_i] = MiniBatchGD(trainX,trainY,GDparams,W,b,lambda,r);
% % W = W_i;
% % b = b_i;
% % costTraining(i) = ComputeCost(trainX, trainY, W, b, lambda);
% % costValidation(i) = ComputeCost(valX, valY, W, b, lambda);
% % GDparams.eta = GDparams.eta*decay_rate;
% % end
% % Conclusion: search e-> 0.7>e>0.01
% % ---------------------------------------------------------------------------------------------
% % ------------------------ Searching the best combination of parameters ---
% % 1)
% % emin = 0.01;
% % emax = 0.7;
% % 3)
% emin = 0.02;
% emax = 0.06;
% % 2)
% % emin = 0.01;
% % emax = 0.1;
% % GDparams.eta = e;
% % GDparams.n_epochs = 10;
% % GDparams.n_batch = 100;
% n_combinations = 50; %75
% % lmax = 0.1;
% % lmin = 0.000001;
% lmax = 1e-05;
% lmin = 1e-06;
% % lmax = 1e-04;
% % lmin = 1e-07;
% etaValues = zeros (1,n_combinations);
% lambdaValues = zeros (1,n_combinations);
% accuracyValues = zeros (1,n_combinations);
% 
% for i = 1 : n_combinations
%     
%     eta = log10(emin) + (log10(emax) - log10(emin))*rand; 
%     GDparams.eta= 10^eta;
%     l = log10(lmin) + (log10(lmax) - log10(lmin))*rand;
%     lambda = 10^l;
%     [W, b, costTraining, costValidation, flag] = FindingParameters(trainX, trainY, valX, valY, GDparams, lambda);
%     
%     if flag == 0
%         etaValues(i) = GDparams.eta;
%         lambdaValues(i)  = lambda;
%        accuracyValues(i) = ComputeAccuracy(valX, valy, W, b); 
%     end
%     if flag == 1
%         etaValues(i) = GDparams.eta;
%         lambdaValues(i)  = lambda;
%        accuracyValues(i) = 0;
%     end
%    
% end
% [bestAcc,I] = sort(accuracyValues);
% % [B,I] = sort(___) also returns a collection of index vectors for any of 
% % the previous syntaxes. I is the same size as A and describes the arrangement 
% % of the elements of A into B along the sorted dimension. For example, if A is a vector, then B = A(I).

% -------------------------------------------------------------------------
%Best set of hyperparamteres.
GDparams.n_epochs = 30;
costTraining = zeros(1, GDparams.n_epochs);
costValidation = zeros(1, GDparams.n_epochs);
decay_rate = 0.95;
r = 0.9;
lambda = 1.8035e-6;
GDparams.eta =0.0265 ;
GDparams.n_batch = 100;
nbatch=100;
for i = 1 : GDparams.n_epochs
costTraining(i) = ComputeCost(trainX, trainY, W, b, lambda);
 costValidation(i) = ComputeCost(valX(:,1:1000), valY(:,1:1000), W, b, lambda);
% costValidation(i) = ComputeCost(valX, valY, W, b, lambda);
    
    [W, b] = MiniBatchGD(trainX, trainY, GDparams, W, b, lambda, r);
    GDparams.eta = decay_rate*GDparams.eta;
end
figure()
plot(1 : GDparams.n_epochs, costTraining, 'r')
hold on
plot(1 : GDparams.n_epochs, costValidation, 'b')
hold off
xlabel('epoch');
ylabel('loss');
legend('training loss', 'validation loss');