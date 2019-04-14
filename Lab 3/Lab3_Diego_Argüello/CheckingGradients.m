close all;
clear all;
addpath cifar-10-matlab/cifar-10-batches-mat; 
[tranX, trainY, trainy] = LoadBatch('data_batch_1.mat');
mean_X = mean(tranX, 2);
Xtr = tranX - repmat(mean_X, [1, size(tranX, 2)]);
m = [50];
std = 0.001;
for i = 1 : size(m, 2) - 1

    W{i} = std*rand(m(i + 1), m(i));
    b{i} = zeros(m(i + 1), 1);
end
batch_size = 100;
delta = 1e-6;
% batch_size = 50;
k = 2;
lambda = 0;
gradCheck(batch_size, tranX, trainY, lambda, W, b, delta, k);