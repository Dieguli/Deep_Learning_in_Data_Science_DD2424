function P = EvaluateClassifier(X, W, b)
[Rx,Cx] = size(X);
[Rw,Cw] = size(W);
b = repmat(b, 1, Cx);  %B= repmat(A,n) returns an array containing n copies of A in the row and column dimensions. 
%The size of B is size(A)*n when A is a matrix.
s = W*X + b;
denorm = repmat(sum(exp(s), 1), Rw, 1);
P = exp(s)./denorm;

end
