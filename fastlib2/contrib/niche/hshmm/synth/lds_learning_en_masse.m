function [lds_params] = lds_learning_en_masse(x_set)
%function [lds_params] = lds_learning_en_masse(x_set)

max_cycles = 100;
tolerance = 1e-4;
K = 2;

n_seq = length(x_set);

lds_params = cell(n_seq, 1);

for i = 1:n_seq
  T_i = size(x_set{i}, 1);
  x_set{i}
  K
  T_i
  max_cycles
  tolerance
  net = lds(x_set{i}, K, T_i, max_cycles, tolerance);
  
  lds_params{i}.A = net.A;
  lds_params{i}.C = net.C;
  lds_params{i}.Q = net.Q;
  lds_params{i}.R = net.R;
  lds_params{i}.mu_0 = net.x0;
  lds_params{i}.Sigma_0 = net.P0;
end
