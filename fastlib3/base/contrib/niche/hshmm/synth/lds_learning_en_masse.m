function [lds_params_set] = lds_learning_en_masse(x_set)
%function [lds_params_set] = lds_learning_en_masse(x_set)

max_cycles = 100;
tolerance = 1e-4;
K = 2;

n_seq = length(x_set);

lds_params_set = cell(n_seq, 1);

for i = 1:n_seq
  T_i = size(x_set{i}, 1);
  x_set{i}
  K
  T_i
  max_cycles
  tolerance
  net = lds(x_set{i}, K, T_i, max_cycles, tolerance);
  
  lds_params_set{i}.A = net.A;
  lds_params_set{i}.C = net.C;
  lds_params_set{i}.Q = net.Q;
  lds_params_set{i}.R = net.R;
  lds_params_set{i}.mu_0 = net.x0;
  lds_params_set{i}.Sigma_0 = net.P0;
end
