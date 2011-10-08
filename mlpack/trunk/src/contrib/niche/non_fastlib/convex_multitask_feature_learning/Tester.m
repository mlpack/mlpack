function test_error = Tester(X_train, Y_train, X_test, Y_test, lambda, epsilon, n_iterations)
%function test_error = Tester(X_train, Y_train, X_test, Y_test, lambda, epsilon, n_iterations)
%
% lambda - regularization parameter on trace norm
% epsilon - perturbation for regularizer
% n_iterations - n_iterations to run alternating algorithm for
% convex multi-task feature learning
% 


% sanity checks
train_size = size(X_train);
test_size = size(X_test);
if ~isequal(train_size([1 3]), test_size([1 3]))
  error(['Either number of dimensions or number of tasks are not' ...
	 ' consistent within X_train and X_task']);
end
  
[n_dims n_training_points n_tasks] = size(X_train);
n_test_points = size(X_test, 2);


[D W] = Learn(X_train, Y_train, lambda, epsilon, n_iterations, 'hinge');


test_error = 0;
for t = 1:n_tasks
  test_error = test_error + ...
      sum((X_test(:,:,t)' * W(:,t)) .* Y_test(:,t) <= 0);
end

test_error = test_error / (n_tasks * n_test_points);
