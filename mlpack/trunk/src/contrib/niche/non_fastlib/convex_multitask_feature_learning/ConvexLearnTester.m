function [] = ConvexLearnTester(X_train, Y_train, X_test, Y_test, ...
				lambda_set, epsilon, n_iterations, results_filename) 
%function [] = ConvexLearnTester(X_train, Y_train, X_test, Y_test, ...
%				 lambda_set, epsilon, n_iterations, results_filename) 
%
% lambda_set - set containing values of regularization parameter on trace norm
% epsilon - perturbation for regularizer
% n_iterations - n_iterations to run alternating algorithm for
% convex multi-task feature learning
% 

if nargin < 8
  error('Wrong number of arguments');
end

if size(lambda_set, 1) < size(lambda_set, 2)
  lambda_set = lambda_set';
end


[n_dims n_training_points n_tasks] = size(X_train);
n_test_points = size(X_test, 2);

n_experiments = length(lambda_set);

training_error = zeros(n_experiments, 1);
test_error = zeros(n_experiments, 1);

for i = 1:length(lambda_set)
  lambda = lambda_set(i);
  [D W] = ConvexLearn(X_train, Y_train, lambda, epsilon, n_iterations, 'hinge');
  
  training_error(i) = ConvexLearnComputeError(X_train, Y_train, W);
  test_error(i) = ConvexLearnComputeError(X_test, Y_test, W);
end


test_error = 0;
for t = 1:n_tasks
  test_error = test_error + ...
      sum((X_test(:,:,t)' * W(:,t)) .* Y_test(:,t) <= 0);
end

test_error = test_error / (n_tasks * n_test_points);

% save it all
save(results_filename, 'lambda_set', 'training_error', 'test_error', ...
     'X_train', 'X_test', 'Y_train', 'Y_test', 'epsilon', ...
     'n_iterations');
