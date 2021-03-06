% function [h, S] = intervalues(X, W, b, k)
function [h, S, mu, v] = hiddenlayers(X, W, b, k, mu_av, v_av)

eps = 0.001;
for i = 1 : k - 1 
    Wi = W{i};
    bi = b{i};
    bi = repmat(bi, 1, size(X, 2));
    s = Wi*X + bi;
    S{i} = s;
    if nargin < 6
        [sbar, mui, vi] = BatchNormForwPass(s, eps);
    else
        [sbar, mui, vi] = BatchNormBackPass(s, eps, mu_av{i}, v_av{i});
    end
    mu{i} = mui;
    v{i} = vi;

    X = max(0, sbar);
    h{i} = X;
      h{i} = max(0,s);
end

end
