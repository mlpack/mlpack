function W = pegasos_lobomargin_trivial_minibatch_bottou(X, Y, n_iterations, n_atoms, W_initial)
%function W = pegasos_lobomargin_trivial_minibatch_bottou(X, Y, n_iterations, n_atoms, W_initial)
%
% X: An element of R^{d \times m}; each column is a point
% Y: An element of {-1,1}^m

n_dims = size(X, 1) / n_atoms;
n_points = size(X, 2);


inds = randperm(n_points);
X = X(:,inds);
Y = Y(inds);

W = W_initial;

cur_ind = 1;

% trivial mini-batch case, when mini_batch_size == 1
for t = 1:n_iterations
  step_size = 1 / sqrt(t);

  if cur_ind == (n_points + 1)
    cur_ind = 1;
    inds = randperm(n_points);
    X = X(:,inds);
    Y = Y(inds);
  end
  
  X_t = X(:,cur_ind);
  Y_t = Y(cur_ind);
  cur_ind = cur_ind + 1;
  
  if Y_t * W' * X_t < 1
    W = W + step_size * Y_t * X_t;
  else
    % do nothing
  end
  
  % project each dictionary atom onto the unit ball
  for j = 1:n_atoms    
    atom_j_inds = ((j - 1) * n_dims + 1):(j * n_dims);
    norm_W_j = norm(W(atom_j_inds));
    if norm_W_j > 1
      W(atom_j_inds) = W(atom_j_inds) / norm_W_j;
    end
  end
end
