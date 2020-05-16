function y = normalizeMu0Sigma1(x)
% Normalizes a dataset such that it has mean 0 and sigma 1.

d = reshape(x, numel(x), 1);

mu = mean(d);

d = d - mu;

sigma = std(d);

y = (x - mu) / sigma;

end

