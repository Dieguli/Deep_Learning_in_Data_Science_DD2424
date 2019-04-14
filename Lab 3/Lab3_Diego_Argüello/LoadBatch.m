function [X, Y, y] = LoadBatch(filename)

indata = load(filename);
X = double(indata.data')/255;   1
y = double(indata.labels') + 1;
K = 10;
N = length(y);
Y = zeros(K, N);
for i = 1 : N
    Y(y(i), i) = 1; 
    %In the position of the number that corresponds to the label we put a 1. The rest will be 0
end                  % convert labels to one-hot representation

end
