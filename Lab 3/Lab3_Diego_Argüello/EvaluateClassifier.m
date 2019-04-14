function P = EvaluateClassifier(H, W, b)

W = W{end};
b = b{end};
X = H{end};
b = repmat(b, 1, size(X, 2));

s = W*X + b;
denorm = repmat(sum(exp(s), 1), size(s, 1), 1);
P = exp(s)./denorm;

end