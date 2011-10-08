function test_error = ConvexComputeTestError(X, Y, W)
%function test_error = ConvexComputeTestError(X, Y, W)

[n_dims n_points n_tasks] = size(X);

test_error = 0;
for t = 1:n_tasks
  test_error = test_error + ...
      sum((X(:,:,t)' * W(:,t)) .* Y(:,t) <= 0);
end

test_error = test_error / (n_tasks * n_points);
