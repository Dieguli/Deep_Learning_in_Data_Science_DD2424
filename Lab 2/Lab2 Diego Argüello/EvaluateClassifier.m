function P = EvaluateClassifier(H, W,b)
% Eq: s1 = W1*x +b1 -> (mx1)=(mxd)*(dx1)+(mx1)
% h = max(0,s1)-> h (mx1)
% Whole batch: (mxn)=(mxd)*(dxn)+(mxn)/ h(mxn)
% s = W2*h+b2-> (Kxn) = (Kxm)*(mxn)+(kxn); p(kxn);
W2 = W{2};
b2 = b{2};
[Rw2,Cw2] = size(W2); 
b2 = repmat(b2, 1, size(H,2));
s = W2*H + b2;
denorm = repmat(sum(exp(s), 1), Rw2, 1);
P = exp(s)./denorm;

end
