function W = svm_lower_bounded_margin(X, y, n_atoms)

n_dims = size(X, 1) / n_atoms;
fprintf('n_dims = %f\n', n_dims);
n_points = length(y);
fprintf('n_points = %f\n', n_points);

cvx_begin
  variable W(n_dims * n_atoms)
  variable xi(n_points)
  minimize sum(xi)
  subject to
    (X * diag(y))' * W >= 1 - xi
    xi >= 0
    for j = 1:n_atoms
      norm(W( ((j - 1) * n_dims + 1):(j * n_dims))) <= 1;
    end
cvx_end
