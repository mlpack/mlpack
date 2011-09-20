function S = UpdateSparseCodes(T, D, lambda, S_initial)

[d k] = size(D);

n = size(T, 2);

% just one sparse code for now


for i = 1:1%n
  s = zeros(k, 1);
  
  Lambda = exp(exp(D * s));
  z = (T ./ Lambda) - ones(d, 1) + D * s;
  regressors = diag(sqrt(Lambda)) * D;
  targets = diag(sqrt(Lambda)) * z;
  
  AtA = regressors' * regressors;
  s_new = l1ls_featuresign(regressors, targets, lambda / 2, s, AtA, ...
			   rank(AtA));
end
S = s_new;
