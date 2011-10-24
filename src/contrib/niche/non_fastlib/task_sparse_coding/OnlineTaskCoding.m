function [W, Z] = OnlineTaskCoding(X, Y, n_atoms, lambda_z)
%function [W, Z] = OnlineTaskCoding(X, Y, n_atoms, lambda_z)
%
% Inputs: all that is necessary, and nothing more..

[n_dims n_points n_tasks] = size(X);

n_iterations = n_tasks

W = normrnd(0, 1, n_dims, n_atoms);
W = normcols(W);

inds = randperm(n_tasks);

cur_ind_set = [];
Z  = [];
for i = 1:n_iterations;
  fprintf('iteration %d\n', i);
  t = inds(i);
  cur_ind_set = [cur_ind_set t]; % inefficient
  
  V = W' * X(:,:,t);
  Z_i = linprogsvm2(V, Y(:,t), lambda_z);
  Z = [Z Z_i]; % inefficient
  
  % given Z, solve SVM with lower bounded margin
  n_rand_tasks = length(cur_ind_set);
  U = zeros(n_dims * n_atoms, n_points * n_rand_tasks);
  for j = 1:n_rand_tasks
    U_j = repmat(X(:,:,cur_ind_set(j)), n_atoms, 1);
    Z_j = Z(:,j);
    U_j = U_j .* repmat(reshape(ones(n_dims, 1) * Z_j', n_dims * ...
				length(Z_j), []), 1, n_points);
    U(:, ((j - 1) * n_points + 1):(j * n_points)) = U_j;
  end
  
  W_vec = svm_lobomargin_block(U, reshape(Y(:,cur_ind_set), ...
					  n_points * n_rand_tasks, []), ...
			       n_atoms, ...
			       reshape(W, n_dims * n_atoms, []));
  W = reshape(W_vec, n_dims, n_atoms);
end
