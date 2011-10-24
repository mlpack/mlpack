function [f, g] = BetaObjective(beta_k, k, eta, phi, X, publishers)
%function [f, g, H] = BetaObjective(beta_k, k, eta, phi, X, publishers)

% compute gradient and Hessian for minimization problem, excluding l_1 regularization term

[V D] = size(X);

P = size(eta, 3);

logsumexp_k_by_publisher = zeros(P, 1);
probs = zeros(V, P);
for p = 1:P
  logsumexp_k_by_publisher(p) = logsumexp(beta_k + eta(:,k,p));
  probs(:,p) = exp(beta_k + eta(:,k,p) - logsumexp_k_by_publisher(p));
end

sum_weighted_counts = zeros(D, 1);
g = zeros(V, 1);
for d = 1:D
  weighted_count = X(:,d) .* phi{d}(:,k);
  sum_weighted_counts(d) = sum(weighted_count);
  g = g - weighted_count + sum_weighted_counts(d) * probs(:,publishers(d));
end


%H = zeros(V, V);

sum_weighted_counts_by_publisher = zeros(P, 1);
for d = 1:D
  p_d = publishers(d);
  sum_weighted_counts_py_publisher(p_d) = ...
      sum_weighted_counts_by_publisher(p_d) + sum_weighted_counts(d);
end

for p = 1:P
%  H = H + sum_weighted_counts_by_publisher(p) ...
%      * (diag(probs(:,p)) - probs(:,p) * probs(:,p)'; % note, this will be a massive, dense V by V matrix, so we should
end


f = 0;
for d = 1:D
  f = f - sum(X(:,d) .* phi{d}(:,k) .* beta_k);
end

for p = 1:P
  f = f + sum_weighted_counts_by_publisher(p) * logsumexp_k_by_publisher(p);
end
