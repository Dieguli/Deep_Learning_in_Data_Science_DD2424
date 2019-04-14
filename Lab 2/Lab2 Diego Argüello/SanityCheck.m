close all;
clear all;
addpath cifar-10-matlab/cifar-10-batches-mat; 
[trainX, trainY, trainy] = LoadBatch('data_batch_1.mat');
mean_trainX = mean(trainX, 2);
trainX = trainX- repmat(mean_trainX, [1, size(trainX, 2)]);
X=trainX(:,1:100);
Y = trainY(:,1:100);
n_epochs = 200;
eta = 0.1;
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
costTraining=zeros(1,n_epochs);
costValidation = zeros(1, n_epochs);
 lambda = 0;
for i = 1:n_epochs
     h = firstLayer(X, W, b);
    P = EvaluateClassifier(h, W, b);
    [grad_W, grad_b] = ComputeGradients(X, Y, P, h, W, b, lambda);
    W{1} = W{1} - eta*grad_W{1};
    W{2} = W{2} - eta*grad_W{2};
    b{1} = b{1} - eta*grad_b{1};
    b{2} = b{2} - eta*grad_b{2};
costTraining(i) = ComputeCost(X,Y, W, b, lambda);
costValidation(i) = ComputeCost(valX, valY, W, b, lambda);
end
% Evaluation: See if it is possible to overfit the training data
figure()
plot(1 : n_epochs, costTraining, 'r')
xlabel('epoch');
ylabel('loss');
hold on


