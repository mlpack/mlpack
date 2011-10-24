function W = TaskCodingWStep(X, Y, Z, lambda_w, W)
%function W = TaskCodingWStep(X, Y, Z, lambda_w, W)
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


minibatch_size = 10;

% pegasos minibatch
%W_vec = pegasos(U, reshape(Y, n_points * n_tasks, []), ...
%		lambda_w, minibatch_size, ...
%		10 * n_points * n_tasks / minibatch_size);

% pegasos minibatch with margin lower bounded by 1
W_vec = pegasos_lobomargin(U, reshape(Y, n_points * n_tasks, []), ...
			   minibatch_size, ...
			   100 * n_points * n_tasks / minibatch_size, n_atoms);

% svm with margin lower bounded by 1
%W_vec = svm_lower_bounded_margin(U, ...
%				 reshape(Y, n_points * n_tasks, []), ...
%				 n_atoms);

%W_vec = svm_lobomargin_block(U, ...
%			     reshape(Y, n_points * n_tasks, []), ...
%			     n_atoms, ...
%			     reshape(W, n_dims * n_atoms, []));

W = reshape(W_vec, n_dims, n_atoms);
