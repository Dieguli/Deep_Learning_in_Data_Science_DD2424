close all;
clear all;
addpath cifar-10-matlab/cifar-10-batches-mat; 
[trainX, trainY, trainy] = LoadBatch('data_batch_1.mat');
[valX, valY, valy] = LoadBatch('data_batch_2.mat');
[testX, testY, testy] = LoadBatch('test_batch.mat');
% % read in training, validation and test data
% [trainX1, trainY1, trainy1] = LoadBatch('data_batch_1.mat');
% [Xtr2, Ytr2, ytr2] = LoadBatch('data_batch_2.mat');
% % [Xtr3, Ytr3, ytr3] = LoadBatch('data_batch_3.mat');
% % [Xtr4, Ytr4, ytr4] = LoadBatch('data_batch_4.mat');
% % [Xtr5, Ytr5, ytr5] = LoadBatch('data_batch_5.mat');
% [Xte, Yte, yte] = LoadBatch('test_batch.mat');
% 
% % regroup training data and validation data
% Xtr = trainX1;
% Ytr = trainY1;
% ytr = trainy1;
% % Xtr = Xtr(:, 1:100);
% % Ytr = Ytr(:, 1:100);
% % ytr = ytr(:, 1:100);
% Xva = Xtr2;
% Yva = Ytr2;
% yva = ytr2;
% 
% % pre-processing
mean_X = mean(trainX, 2);
trainX = trainX - repmat(mean_X, [1, size(trainX, 2)]);
valX = valX - repmat(mean_X, [1, size(valX, 2)]);
testX = testX - repmat(mean_X, [1, size(testX, 2)]);

% initialization
k = 3;
m = [50 30];
% k = 2;
% m = 50;
 n_epochs = 50;
%  n_epochs = 10; %Ej 4
n_batch = 100;
Eta = [];
Lambda = [];
acc_va = [];
%SEARCH HYPER-PARAMETERS
% n_pairs = 50;
%  n_epochs = 10;
% % eta_max = 0.9;
% % eta_min = 0.028;
% % lambda_max = 5.3e-3;
% % lambda_min = 1e-7;
% % 1st interval:
% % eta_max = 0.7;
% % eta_min = 0.01;
% % lambda_max = 1e-2;
% % lambda_min = 1e-6;
% %2nd interval
% % eta_max = 0.2;
% % eta_min = 0.08;
% % lambda_max = 3e-3;
% % lambda_min = 4e-5;
% %3rd interval
% eta_max = 0.09;
% eta_min = 0.07;
% lambda_max = 3e-4;
% lambda_min = 5e-5;
% for i = 1 : n_pairs
%     %randomly generate eta and lambda
%     e = log10(eta_min) + (log10(eta_max) - log10(eta_min))*rand(1, 1); 
%     eta = 10^e;
%     e = log10(lambda_min) + (log10(lambda_max) - log10(lambda_min))*rand(1, 1);
%     lambda = 10^e;
%     
%     GDparams = setGDparams(n_batch, eta, n_epochs);
%     tic
%     [W, b, Jtr, Jva, flag, mu_av, v_av] = main(Xtr, Ytr, Xva, Yva, GDparams, lambda, k, m);
%     toc
%     
%     if flag == 0
%         Eta = [Eta, eta];
%         Lambda = [Lambda, lambda];
%         acc_va = [acc_va, ComputeAccuracy(Xva, yva, W, b, k, mu_av, v_av)];
%     end
%     disp(['i = ' num2str(i) ', test accuracy = ' num2str(acc_va(i)*100) '%'])
% end
%TRAINING
% eta = 0.4380;
% lambda = 1.0907e-4;
% eta = 0.1;
% lambda = 1e-4;
 eta = 0.0864; %eta optimum+ medium
% eta = 0.21;% eta = 0.2; % eta = 0.1; %eta training
% eta = 0.001; %eta low
% eta = 0.2; %eta high: higher does not get any result
% bn : Elapsed time is 47.572261 seconds.
% training accuracy:62.68%
% test accuracy:43.19%
lambda = 1.8245e-4;
GDparams = setGDparams(n_batch, eta, n_epochs);
%  [W, b, Jtr, Jva, ~, mu_av, v_av] = training(Xtr, Ytr, Xva, Yva, GDparams, lambda, k, m);
 [W, b, Jtr, Jva, ~] = training(Xtr, Ytr, valX, Yva, GDparams, lambda, k, m);
% acc_tr = ComputeAccuracy(Xtr, ytr, W, b, k, mu_av, v_av);
 acc_tr = ComputeAccuracy(Xtr, ytr, W, b, k);
disp(['training accuracy:' num2str(acc_tr*100) '%'])
% acc_te = ComputeAccuracy(Xte, yte, W, b, k, mu_av, v_av);
acc_te = ComputeAccuracy(testX, yte, W, b, k);
disp(['test accuracy:' num2str(acc_te*100) '%'])

figure()
plot(1 : GDparams.n_epochs, Jtr, 'b')
hold on
plot(1 : GDparams.n_epochs, Jva, 'r')
hold off
xlabel('epoch');
ylabel('loss');
legend('training loss', 'validation loss');

