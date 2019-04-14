close all;
clear all;
addpath cifar-10-matlab/cifar-10-batches-mat; 
[trainX, trainY, trainy] = LoadBatch('data_batch_1.mat');
mean_trainX = mean(trainX, 2);
trainX = trainX- repmat(mean_trainX, [1, size(trainX, 2)]);
% nbatch = 10;
 nbatch = 20;
% nbatch = 30;
lambda = 0;
d = size(trainX,1);
K = size (trainY,1);
k= 3;
m = [50 30];
epochs = 50;
sigma = 0.001;
mu = 0;
% % [W,b] = init_para(m,sigma);
% W = {};
% b = {};
h = 1e-5;
% W{1} = mu+ sigma*rand(m(1), d);
% b{1} =  zeros(m(1), 1);
% W{k+1} = mu+ sigma*rand(K, m(k));
% b{k+1} =  zeros(K, 1);
% if k==1
%   W{2} = 
%   b{2} = zeros(m, 1);
% else
%     
% for i = 1 : k
%   W{}  
% end
% for i = 1 : size(m, 2) - 1
%     
%     W{i} =mu+ sigma*rand(m(i + 1), m(i));
%     b{i} = zeros(m(i + 1), 1);
% end
arg = [d, m, K];
[W, b] = initialize(arg, sigma);
gradCheck(nbatch, trainX, trainY, lambda, W, b, h, k);