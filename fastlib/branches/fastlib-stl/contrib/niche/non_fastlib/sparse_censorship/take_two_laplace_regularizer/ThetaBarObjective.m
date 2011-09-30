function [f, g] = ThetaBarObjective(theta_bar, theta, phi, X, counts_by_doc) 
%function [f, g, H] = ThetaBarObjective(theta_bar, theta, phi, X, counts_by_doc) 

% compute gradient and Hessian for minimization problem, excluding l_1 regularization term

K = length(theta_bar);

D = length(phi);


sum_phi_d_n = zeros(K, 1);
for d = 1:D
  sum_phi_d_n = sum_phi_d_n + phi{d}' * X(:,d);
end

f = -dot(theta_bar, sum_phi_d_n);
g = zeros(K, 1);
%H = zeros(K, K);
for d = 1:D
  f = f + counts_by_doc(d) * logsumexp(theta_bar + theta(:,d));

  log_sum_exp_theta_d = logsumexp(theta_bar + theta(:,d));
  theta_d_probs = exp(theta_bar + theta(:,d) - repmat(log_sum_exp_theta_d, K, 1));

  a = g ...
      - phi{d}' * X(:,d) ...
      + counts_by_doc(d) * theta_d_probs;
  
  g(:,1) = a;
  

  % just as in ThetaObjective.m, we again my run into the problem
  % that the Hessian is always rank k-1. We should check to see if
  % this is indeed the case here as well.
%  H = H + ...
%      counts_by_doc(d) * ...
%      (diag(theta_d_probs) - theta_d_probs * theta_d_probs');
end
