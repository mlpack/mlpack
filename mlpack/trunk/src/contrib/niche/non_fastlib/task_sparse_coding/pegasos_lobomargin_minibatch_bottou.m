function W = pegasos_lobomargin_minibatch_bottou(X, Y, minibatch_size, n_iterations, n_atoms, W_initial)
%function W = pegasos_lobomargin_minibatch_bottou(X, Y, minibatch_size, n_iterations, n_atoms, W_initial)
%
% X: An element of R^{d \times m}; each column is a point
% Y: An element of {-1,1}^m
%
% Sampling without replacement by:
%   1) Exhausting all examples sequentially from a random permutation
%   2) Creating a new random permutation
%   3) Repeating

[n_dims_full n_points] = size(X);
n_dims = n_dims_full / n_atoms;

inds = randperm(n_points);
X = X(:,inds); % perhaps its faster to reorder the data now rather
               % than picking random columns later
Y = Y(inds);

W = W_initial;

cur_ind = 1;

for t = 1:n_iterations
  step_size = 1 / sqrt(t);

  if cur_ind == (n_points + 1)
    cur_ind = 1;
    inds = randperm(n_points);
    X = X(:,inds);
    Y = Y(inds);
  end
  
  draws = cur_ind:min((cur_ind + minibatch_size - 1), n_points);
  cur_ind = draws(end) + 1;
  
  subgrad = zeros(n_dims_full, 1);
  
  % slow subgradient computation, shown for code readability
  %for i = 1:length(draws)
  %  X_t = X(:,draws(i));
  %  Y_t = Y(draws(i));
  %  
  %  if Y_t * W' * X_t < 1
  %    subgrad = subgrad + Y_t * X_t;
  %  end
  %end
  
  % fast subgradient computation
  margin_error_inds = draws(find(Y(draws)' .* (W' * X(:,draws)) < 1));
  subgrad = subgrad + ...
	    sum(bsxfun(@times, X(:,margin_error_inds), Y(margin_error_inds)'), 2);
  
  W = W + (step_size / minibatch_size) * subgrad;  

  % project each dictionary atom onto the unit ball
  for j = 1:n_atoms    
    atom_j_inds = ((j - 1) * n_dims + 1):(j * n_dims);
    norm_W_j = norm(W(atom_j_inds));
    if norm_W_j > 1
      W(atom_j_inds) = W(atom_j_inds) / norm_W_j;
    end
  end
end
