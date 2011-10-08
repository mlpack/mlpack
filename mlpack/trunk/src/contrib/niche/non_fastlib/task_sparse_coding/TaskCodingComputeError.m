function test_error = TaskCodingComputeError(X, Y, W, Z)
%function test_error = TaskCodingComputeError(X, Y, W, Z)

[n_dims n_points n_tasks] = size(X);

obj = 0;
for t = 1:n_tasks
  obj = obj + sum(((X(:,:,t)' * (W * Z(:,t))) .* Y(:,t) <= 0));
end

test_error = obj / (n_tasks * n_points);
