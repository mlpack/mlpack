function [f, g, H] = ThetaObjective(theta_d, phi_d, X_d, counts_d) 

% compute gradient and Hessian for minimization problem, excluding l_1 regularization term

K = length(theta_d);

log_sum_exp_theta_d = logsumexp(theta_d);

theta_d_probs = exp(theta_d - repmat(log_sum_exp_theta_d, K, 1));

f = -dot(theta_d, phi_d' * X_d) ...
    + counts_d * logsumexp(theta_d)

g = -phi_d' * X_d ...
    + counts_d * theta_d_probs;

% is this a problem? it looks like the Hessian is always rank k-1, probably due to the one degree of freedom
% (without regularization, solutions are equivalent up to translation by a constant times the ones vector)
H = counts_d * ...
    (diag(theta_d_probs) - theta_d_probs * theta_d_probs');
