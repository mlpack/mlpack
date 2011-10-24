function [] = TaskCodingTester(X_train, Y_train, X_test, Y_test, ...
			       n_atoms_set, lambda_w, lambda_z_set, ...
			       W_true, Z_true, results_dirname)
%function [] = TaskCodingTester(X_train, Y_train, X_test, Y_test, ...
%			       n_atoms_set, lambda_w, lambda_z_set, ...
%			       W_true, Z_true, results_dirname)

if nargin < 10
  error('Wrong number of arguments');
end

problem_filepath = [results_dirname '/problem.mat'];
results_filepath = [results_dirname '/results.mat'];

if ~isempty(dir(results_dirname))
  if ~isempty(dir(problem_filepath)) || ~isempty(dir(results_filepath))
    error(['Results directory ''%s'' already exists and contains' ...
	   ' results.mat or problem.mat\n'], ...
	  results_dirname);
  end
else
  mkdir(results_dirname);
end

if size(lambda_z_set, 1) < size(lambda_z_set, 2)
  lambda_z_set = lambda_z_set';
end

n_experiments = length(n_atoms_set) * length(lambda_z_set);

if size(n_atoms_set, 1) > size(n_atoms_set, 2)
  n_atoms_set = n_atoms_set';
end

if size(lambda_z_set, 1) > size(lambda_z_set, 2)
  lambda_z_set = lambda_z_set';
end

results = zeros(n_experiments, 6);
results(:,1) = reshape(ones(length(lambda_z_set), 1) * n_atoms_set, ...
		       n_experiments, 1);
results(:,2) = repmat(lambda_z_set', length(n_atoms_set), 1);
results(:,3:6) = -1; % write negative ones to indicate pending results
%opt_training_error = zeros(n_experiments, 1);
%opt_test_error = zeros(n_experiments, 1);
%learned_training_error = zeros(n_experiments, 1);
%learned_test_error = zeros(n_experiments, 1);



% save it all
save(problem_filepath, ...
     'X_train', 'X_test', 'Y_train', 'Y_test', 'W_true', 'Z_true');

for i = 1:n_experiments
  n_atoms = results(i,1);
  lambda_z = results(i,2);

  % using optimal dictionary
  Z = TaskCodingZStep(X_train, Y_train, W_true, lambda_z);
  results(i,3) = TaskCodingComputeError(X_train, Y_train, W_true, Z);
  results(i,4) = TaskCodingComputeError(X_test, Y_test, W_true, Z);
  
  % using learned dictionary
  [W Z] = TaskCoding(X_train, Y_train, n_atoms, lambda_w, lambda_z);
  results(i,5) = TaskCodingComputeError(X_train, Y_train, W, Z);
  results(i,6) = TaskCodingComputeError(X_test, Y_test, W, Z);

  % save it all
  if i > 1
    movefile(results_filepath, [results_dirname '/penultimate_results.mat']);
  end
  save(results_filepath, 'results', 'lambda_w');  
end
