function [A, C, Q, R, mu_0, Sigma_0] = ...
    init_params(n_dims_latent, n_dims_obs)
%function [A, C, Q, R, mu_0, Sigma_0] = ...
%    init_params(n_dims_latent, n_dims_obs)
%
% n_dims_latent defaults to 2
% n_dims_obs defaults to 1


if ~exist('n_dims_latent')
  n_dims_latent = 2;
  fprintf(['parameter ''n_dims_latent'' not passed, setting it to ' ...
	   '%d\n'], n_dims_latent);
end

if ~exist('n_dims_obs')
  n_dims_obs = 1;
  fprintf(['parameter ''n_dims_obs'' not passed, setting it to ' ...
	   '%d\n'], n_dims_obs);
end


mu_0 = rand(n_dims_latent, 1);
Sigma_0 = random_cov(n_dims_latent);

A = rand(n_dims_latent);
A = A / max(eig(A));

C = rand(n_dims_obs, n_dims_latent);

Q = diag(rand(n_dims_latent,1));%random_cov(n);

R = random_cov(n_dims_obs);
