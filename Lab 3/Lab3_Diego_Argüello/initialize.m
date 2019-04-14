function [W, b] = initialize(m, sigma)

for i = 1 : size(m, 2) - 1
    
    W{i} = sigma*rand(m(i + 1), m(i));
    b{i} = zeros(m(i + 1), 1);
end

end
