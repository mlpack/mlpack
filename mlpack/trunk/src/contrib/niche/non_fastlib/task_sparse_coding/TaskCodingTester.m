function [] = TaskCodingTester(X_train, X_test, Y_train, Y_test, ...
			       n_atoms, lambda_w, lambda_z_set, ...
			       W_true, results_filename)
%function [] = TaskCodingTester(X_train, X_test, Y_train, Y_test, ...
%			       n_atoms, lambda_w, lambda_z_set, ...
%			       W_true, results_filename)


if nargin < 9
  error('Wrong number of arguments');
end

if size(lambda_z_set, 1) < size(lambda_z_set, 2)
  lambda_z_set = lambda_z_set';
end

n_experiments = length(lambda_z_set);

opt_training_error = zeros(n_experiments, 1);
opt_test_error = zeros(n_experiments, 1);
learned_training_error = zeros(n_experiments, 1);
learned_test_error = zeros(n_experiments, 1);

for i = 1:length(lambda_z_set)
  lambda_z = lambda_z_set(i);

  % using optimal dictionary
  Z = TaskCodingZStep(X_train, Y_train, W_true, lambda_z);
  opt_training_error(i) = TaskCodingComputeError(X_train, Y_train, W_true, Z);
  opt_test_error(i) = TaskCodingComputeError(X_test, Y_test, W_true, Z);
  
  % using learned dictionary
  [W Z] = TaskCoding(X_train, Y_train, n_atoms, lambda_w, lambda_z);
  learned_training_error(i) = TaskCodingComputeError(X_train, Y_train, W, Z);
  learned_test_error(i) = TaskCodingComputeError(X_test, Y_test, W, Z);
end

% save it all
save(results_filename, 'lambda_z_set', 'opt_training_error', ...
     'opt_test_error', 'learned_training_error', 'learned_test_error', ...
     'X_train', 'X_test', 'Y_train', 'Y_test', 'n_atoms', 'lambda_w', ...
     'W_true');
