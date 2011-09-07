function [f, g, H] = EtaObjective(beta_k, eta_k_p, k, p, phi, beta_k, X, publishers)

% compute gradient and Hessian for minimization problem, excluding l_1 regularization term

[V D] = size(X);

P = size(eta, 3);

probs= exp(beta_k + eta_k_p - logsumexp(beta_k + eta_k_p));

f = ???;

sum_weighted_counts = zeros(d, 1);
g = zeros(V, 1);
for d = find(publishers == p)
  weighted_count = X(:,d) .* phi{d}(:,k);
  sum_weighted_counts(d) = sum(weighted_count);
  g = g - weighted_count + sum_weighted_counts(d) * probs;
end

H = sum(sum_weighted_counts) ...
    * (diag(probs) - probs * probs'; % note, this will be a massive, dense V by V matrix, so we should not create this explicitly!
