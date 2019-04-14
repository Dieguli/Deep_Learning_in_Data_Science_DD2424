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
% use 100 elements of the data, lambda=0,  epochs=200, eta = 0.1 ("reasonable")
GDparams.n_epochs = 5;
GDparams.n_batch = 100;
GDparams.eta = 0.01;
lambda = 0;
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
costTraining=zeros(1,GDparams.n_epochs);
costValidation = zeros(1, GDparams.n_epochs);
% r = 0.5;
decay_rate = 0.95;
% r = 0.9;
r = 0.99;
% Training
for i = 1:GDparams.n_epochs
%     if i==1
%  costTraining(i) = ComputeCost(trainX, trainY, W, b, lambda);
%     end
% [Wi,bi] = MiniBatchGD_old(trainX,trainY,GDparams,W,b,lambda);
[Wi,bi] = MiniBatchGD(trainX,trainY,GDparams,W,b,lambda,r);
W = Wi;
b = bi;
costTraining(i) = ComputeCost(trainX, trainY, W, b, lambda);
costValidation(i) = ComputeCost(valX, valY, W, b, lambda);
GDparams.eta = GDparams.eta*decay_rate;
end
% Evaluation: See if it is possible to overfit the training data
figure()
plot(1 : GDparams.n_epochs, costTraining, 'r')
hold on
plot(1 : GDparams.n_epochs, costValidation, 'b')
hold off
xlabel('epoch');
ylabel('loss');
legend('training loss', 'validation loss');