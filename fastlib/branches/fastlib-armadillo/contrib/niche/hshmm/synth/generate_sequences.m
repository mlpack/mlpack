function [q_set x_set] = ...
    generate_sequences(n_seq, n_dims_latent, n_dims_obs)
%function [q_set x_set] = ...
%    generate_sequences(n_seq, n_dims_latent, n_dims_obs)

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


[A, C, Q, R, mu_0, Sigma_0] = ...
    init_params(n_dims_latent, n_dims_obs);

q_set = cell(n_seq, 1);
x_set = cell(n_seq, 1);

for i = 1:n_seq
  [q_set{i} x_set{i}] = generate_data_from_lds(A, C, Q, R, mu_0, Sigma_0, 70);
end
