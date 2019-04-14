function acc = ComputeAccuracy(X, y, W, b)
% The accuracy of a classifier for a given set of examples is the
% percentage of examples for which it gets the correct answer
P = EvaluateClassifier(X, W, b);
[K,N] = size(P);
k = zeros(1,N);
for i=1:N
    [M, I]= max(P(:,i));
    k(i) = I;
end
acc = length(find(y - k == 0))/length(y); % caculate the percentage 

end

