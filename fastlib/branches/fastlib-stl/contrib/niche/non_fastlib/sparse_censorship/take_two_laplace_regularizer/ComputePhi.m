function phi = ComputePhi(theta, beta, eta, publishers)

D = length(publishers);
[V K] = Size(beta, 1);

phi = cell(D, 1); % matlab doesn't do sparse tensors, so we use a
                  % cell-array of sparse matrices
for d = 1:D
  phi{d} = sparse(V, K);
end


log_sum_exp_beta = logsumexp(beta);
for d = 1:D
  p = publishers(d);
  log_sum_exp_theta_d = logsumexp(theta(:,d));
  for k = 1:K
    for v = inds_by_doc{d}
      % using logsumexp from Lightspeed, make sure it's in the path
      phi{d}(v,k) = ...
          beta(v,k) + eta(v,k,p) - log_sum_exp_beta(k) ...
          + theta(k,d) - log_sum_exp_theta_d;
    end
  end
  phi{d}(inds_by_doc{d},:) = ...
      exp(phi{d}(inds_by_doc{d},:) ...
          - repmat(logsumexp(phi{d}(inds_by_doc{d},:), 2), 1, K));
end
