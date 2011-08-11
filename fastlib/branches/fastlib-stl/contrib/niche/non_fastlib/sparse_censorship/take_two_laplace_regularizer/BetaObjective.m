function [f, g, H] = BetaObjective(beta_k, k, eta, phi, beta_k, X, publishers)

% compute gradient and Hessian for minimization problem, excluding l_1 regularization term

[V D] = size(X);

P = size(eta, 3);

probs = zeros(V, P);
for p = 1:P
  probs(:,p) = exp(beta_k + eta(:,k,p) - logsumexp(beta_k + eta(:,k,p)));
end

sum_weighted_counts = zeros(d, 1);
g = zeros(V, 1);
for d = 1:D
  weighted_count = X(:,d) .* phi{d}(:,k);
  sum_weighted_counts(d) = sum(weighted_count);
  g = g + weighted_count - sum_weighted_counts(d) * probs(:,publishers(d))
end

H = zeros(V, V);
for d = 1:D
  p = publishers(d);
  H = H + sum_weighted_counts(d) ...
      * diag(probs(:,d) + probs(:,d) * probs(:,d)'; % note, this will be a massive, dense V by V matrix, so we should not create this explicitly!
end

