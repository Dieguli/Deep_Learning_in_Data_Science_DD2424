close all;
clear all;
addpath cifar-10-matlab/cifar-10-batches-mat; 
% Dimensions: X(dxN), Y(KxN), W(Kxd), b(Kx1), P(KxN), y(1xN) with K = 10, N = 1000
% and d = 3072
% 1) Read in and store the training, validation and test data.
[trainX, trainY, trainy] = LoadBatch('data_batch_1.mat');
[valX, valY, valy] = LoadBatch('data_batch_2.mat');
[testX, testY, testy] = LoadBatch('test_batch.mat');
% 2)Initalization of the parameters W and b.
% Initialize each entry to have Gaussian random values with zero mean and standard deviation .01
mu = 0;
sigma = 0.01;
K = size(trainY, 1);
d = size(trainX, 1);
W = mu + randn(K, d)*sigma;
b = mu + randn(K, 1)*sigma;
lambda = 0;
% 3)check  analytical gradient is correct
batch_size = 100;
[numericalGradient_b, numericalGradient_W] = ComputeGradsNumSlow(trainX(:, 1 : batch_size),trainY(:, 1 : batch_size), W, b, lambda, 1e-6);
% [ngrad_b, ngrad_W] = ComputeGradsNum(trainX(:, 1 : batch_size),trainY(:, 1 : batch_size), W, b, lambda, 1e-6);
P = EvaluateClassifier(trainX(:, 1 : batch_size), W, b);
[gradient_W, gradient_b] = ComputeGradients(trainX(:, 1 : batch_size),trainY(:, 1 : batch_size), P, W, lambda);
relativeError_b = max(abs(numericalGradient_b - gradient_b)./max(0, abs(numericalGradient_b) + abs(gradient_b)))
relativeError_W = max(max(abs(numericalGradient_W - gradient_W)./max(0, abs(numericalGradient_W) + abs(gradient_W))))