close all;
clear all;
addpath cifar-10-matlab/cifar-10-batches-mat; 
% Dimensions: X(dxN), Y(KxN), W(Kxd), b(Kx1), P(KxN), y(1xN) with K = 10, N = 1000
% and d = 3072
% 1) Read in and store the training, validation and test data.
[trainX, trainY, trainy] = LoadBatch('data_batch_1.mat');
[valX, valY, valy] = LoadBatch('data_batch_2.mat');
[testX, testY, testy] = LoadBatch('test_batch.mat');
d = size(trainX, 1);
K = size(trainY, 1);
m = 50;
[W,b] = Initialization(m,d,K);
elementsBatch = 10;
lambda = 0.01;
H = firstLayer(trainX(:,1:elementsBatch),W,b);
P = EvaluateClassifier(H, W,b);
h =  1e-5;% best precission 
[ngrad_b, ngrad_W] = ComputeGradsNumSlow(trainX(:, 1 : elementsBatch),trainY(:, 1 : elementsBatch), W, b, lambda,h); 
[gradW, gradb] = ComputeGradients(trainX(:, 1 : elementsBatch),trainY(:, 1 : elementsBatch), P, H, W, b, lambda);
agrad_b1 = gradb{1};
agrad_b2 = gradb{2};
agrad_W1 = gradW{1};
agrad_W2 = gradW{2};
% relative error rate
eps = 0; 
errorb1 = sum(abs(ngrad_b{1} - agrad_b1)/max(eps, sum(abs(ngrad_b{1}) + abs(agrad_b1))))
errorW1 = sum(sum(abs(ngrad_W{1} - agrad_W1)/max(eps, sum(sum(abs(ngrad_W{1}) + abs(agrad_W1))))))
errorb2 = sum(abs(ngrad_b{2} - agrad_b2)/max(eps, sum(abs(ngrad_b{2}) + abs(agrad_b2))))
errorW2 = sum(sum(abs(ngrad_W{2} - agrad_W2)/max(eps, sum(sum(abs(ngrad_W{2}) + abs(agrad_W2))))))
