function [f, g] = EtaObjective(eta_k_p, beta_k, k, p, phi, X, publishers)
%function [f, g, H] = EtaObjective(eta_k_p, beta_k, k, p, phi, X, publishers)

% compute gradient and Hessian for minimization problem, excluding l_1 regularization term

[V D] = size(X);

logsumexp_k_p = logsumexp(beta_k + eta_k_p);
probs = exp(beta_k + eta_k_p - logsumexp_k_p);

sum_weighted_counts = zeros(D, 1);
g = zeros(V, 1);
for d = find(publishers == p)
  weighted_count = X(:,d) .* phi{d}(:,k);
  sum_weighted_counts(d) = sum(weighted_count);
  g = g - weighted_count + sum_weighted_counts(d) * probs;
end

%H = sum(sum_weighted_counts) ...
%    * (diag(probs) - probs * probs'; % note, this will be a massive, dense V by V matrix, so we should not create this explicitly!


f = 0;
for d = find(publishers == p)
  f = f - sum(X(:,d) .* phi{d}(:,k) .* eta_k_p);
end

f = f + sum(sum_weighted_counts) * logsumexp_k_p;
