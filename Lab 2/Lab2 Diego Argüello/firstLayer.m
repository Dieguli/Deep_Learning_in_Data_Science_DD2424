function H= firstLayer(X, W,b)
[Rx,Cx] = size(X);
W1 = W{1};
b1 = b {1};
b1 = repmat(b1, 1,Cx);
H = W1*X + b1;
H = max(0,H);
end


