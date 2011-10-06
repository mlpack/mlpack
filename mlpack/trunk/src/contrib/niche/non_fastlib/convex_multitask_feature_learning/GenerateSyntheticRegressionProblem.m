function [X,Y,U,A] = GenerateSyntheticRegressionProblem(n_dims, n_points, n_tasks, n_dims_subspace)
%function [X,Y,U,A] = GenerateSyntheticRegressionProblem(n_dims, n_points, n_tasks, n_dims_subspace)

% generate the feature inducers
U = orth(normrnd(0, 1, n_dims, n_dims));

% generate linear maps from features to outputs
A = normrnd(0, 1, n_dims, n_tasks);
% only the first n_dims_subspace features will be relevant
A((n_dims_subspace + 1):end, :) = 0;

% generate data
X = normrnd(0,1, n_dims, n_points, n_tasks);
Y = zeros(n_points, n_tasks);
for t = 1:n_tasks
  Y(:,t) = X(:,:,t)' * U * A(:,t);
end
