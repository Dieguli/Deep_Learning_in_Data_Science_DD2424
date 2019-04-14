function [X,Y,y] = LoadBatch(fname)
%LOADBATCH: With this function it is possible to store in X,Y and y the
%images and their labels
%A load the file and creates a strcut with the fileds: data (1000x3072), labels (1000x1) and
%batch_label. 
%For the names and size of the variables, the noation of the assignment
%description has been followed
A = load(fname);
X = double(A.data')/255;   % divide by 255 in order to have 0-1 values.          
y = double(A.labels') + 1;  % encode the labels between 1-10
K = 10;
N = length(y);
Y = zeros(K, N);
% change labels into the one-hot representation
for i = 1 : N
    Y(y(i), i) = 1; 
    %In the position of the number that corresponds to the label we put a 1. The rest will be 0
end
 

end



