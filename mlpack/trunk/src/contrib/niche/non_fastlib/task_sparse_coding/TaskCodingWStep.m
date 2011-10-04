function W = TaskCodingWStep(X, Y, Z, lambda_w)
%function W = TaskCodingWStep(X, Y, Z, lambda_w)
%

[n_dims n_points n_tasks] = size(X);
n_atoms = size(Z, 1);

U = zeros(n_dims * n_atoms, n_points * n_tasks);
for t = 1:n_tasks
  U_t = repmat(X(:,:,t), n_atoms, 1);
  Z_t = Z(:,t);
  U_t = U_t .* repmat(reshape(ones(n_dims, 1) * Z_t', n_dims * length(Z_t), []), ...
		      1, n_points);
  U(:, ((t - 1) * n_points + 1):(t * n_points)) = U_t;
end

W_vec = pegasos(U, reshape(Y, n_points * n_tasks, []), lambda_w, 1,  1000);
W = reshape(W_vec, n_dims, n_atoms);
